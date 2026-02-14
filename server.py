# Modified from original MCP-Markdown-RAG by Zackriya Solutions
# (https://github.com/Zackriya-Solutions/MCP-Markdown-RAG)
# Changes: multi-provider embedding support (Gemini/Voyage/OpenAI),
#          configurable Milvus address, batch embedding for API limits.

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

from fastmcp import FastMCP
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.text_splitter import TokenTextSplitter
from pydantic import Field
from pymilvus import MilvusClient, model

from utils import (
    COLLECTION_NAME,
    Entity,
    INDEX_DATA_PATH,
    SearchResult,
    ensure_collection,
    get_changed_files,
    list_md_files,
    update_tracking_file,
)

mcp = FastMCP(
    "mcp-markdown-rag",
    instructions="""This MCP server provides semantic search capabilities over markdown files using 
    vector embeddings and Milvus database. It enables you to index markdown documents and perform 
    intelligent searches to find relevant content based on semantic similarity rather than just 
    keyword matching.""",
)


if not os.path.exists(INDEX_DATA_PATH):
    os.makedirs(INDEX_DATA_PATH)

# --- Embedding Provider Selection ---
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
embedding_log_details = ""


class VertexEmbeddingFunction:
    """Vertex AI native embedding function using publishers/google/models/*:predict."""

    def __init__(self, model_name: str, project: str | None, location: str, dimensions: int):
        from google.auth import default
        from google.auth.transport.requests import Request

        self.model_name = model_name
        self.location = location
        self.dimensions = dimensions if dimensions > 0 else None
        self._request = Request()
        self._credentials, detected_project = default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.project = project or detected_project
        if not self.project:
            raise ValueError("VERTEX_PROJECT is required for EMBEDDING_PROVIDER=vertex")
        self.endpoint = (
            f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project}"
            f"/locations/{self.location}/publishers/google/models/{self.model_name}:predict"
        )

    def _ensure_access_token(self) -> str:
        should_refresh = (
            not self._credentials.valid
            or not self._credentials.token
            or self._credentials.expired
        )
        if not should_refresh and self._credentials.expiry:
            threshold_ts = (datetime.now(timezone.utc) + timedelta(seconds=60)).timestamp()
            expiry = self._credentials.expiry
            if expiry.tzinfo is None:
                expiry_ts = expiry.replace(tzinfo=timezone.utc).timestamp()
            else:
                expiry_ts = expiry.timestamp()
            should_refresh = expiry_ts <= threshold_ts
        if should_refresh:
            self._credentials.refresh(self._request)
        return str(self._credentials.token)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {"instances": [{"content": text} for text in texts]}
        if self.dimensions:
            payload["parameters"] = {"outputDimensionality": self.dimensions}

        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._ensure_access_token()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"[Vertex] Embedding request failed ({exc.code}): {detail}"
            ) from exc

        result = json.loads(body)
        predictions = result.get("predictions", [])
        if not isinstance(predictions, list):
            raise ValueError("[Vertex] Invalid response: missing predictions array")
        if len(predictions) != len(texts):
            raise ValueError(
                f"[Vertex] Embedding count mismatch: expected {len(texts)}, got {len(predictions)}"
            )

        vectors: list[list[float]] = []
        for idx, prediction in enumerate(predictions):
            embeddings = prediction.get("embeddings", {}) if isinstance(prediction, dict) else {}
            values = embeddings.get("values") if isinstance(embeddings, dict) else None
            if not isinstance(values, list):
                raise ValueError(
                    f"[Vertex] Invalid response shape at predictions[{idx}].embeddings.values"
                )
            vectors.append(values)
        return vectors

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_batch(texts)

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed_batch(texts)


if EMBEDDING_PROVIDER == "gemini":
    from pymilvus.model.dense import OpenAIEmbeddingFunction
    model_name = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    embedding_fn = OpenAIEmbeddingFunction(
        model_name=model_name,
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        dimensions=EMBEDDING_DIM,
    )
    embedding_log_details = (
        f"provider=gemini model={model_name} "
        "base_url=https://generativelanguage.googleapis.com/v1beta/openai/"
    )
elif EMBEDDING_PROVIDER == "voyage":
    from pymilvus.model.dense import VoyageEmbeddingFunction
    model_name = os.getenv("EMBEDDING_MODEL", "voyage-3")
    embedding_fn = VoyageEmbeddingFunction(
        model_name=model_name,
        api_key=os.getenv("VOYAGE_API_KEY"),
    )
    embedding_log_details = f"provider=voyage model={model_name}"
elif EMBEDDING_PROVIDER == "openai":
    from pymilvus.model.dense import OpenAIEmbeddingFunction
    model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_fn = OpenAIEmbeddingFunction(
        model_name=model_name,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    embedding_log_details = f"provider=openai model={model_name}"
elif EMBEDDING_PROVIDER == "openai-compatible":
    from pymilvus.model.dense import OpenAIEmbeddingFunction
    model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    base_url = os.getenv("EMBEDDING_BASE_URL")

    embedding_fn = OpenAIEmbeddingFunction(
        model_name=model_name,
        api_key=os.getenv("EMBEDDING_API_KEY"),
        base_url=base_url,
        dimensions=EMBEDDING_DIM,
    )
    embedding_log_details = f"provider=openai-compatible model={model_name} base_url={base_url}"
elif EMBEDDING_PROVIDER == "vertex":
    model_name = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    location = os.getenv("VERTEX_LOCATION", "us-central1")
    embedding_fn = VertexEmbeddingFunction(
        model_name=model_name,
        project=os.getenv("VERTEX_PROJECT"),
        location=location,
        dimensions=EMBEDDING_DIM,
    )
    embedding_log_details = (
        f"provider=vertex model={model_name} project={embedding_fn.project} "
        f"location={location} endpoint={embedding_fn.endpoint}"
    )
else:
    embedding_fn = model.DefaultEmbeddingFunction()  # 기본 로컬 (768d)
    embedding_log_details = "provider=local model=DefaultEmbeddingFunction"

print(f"[Embedding] {embedding_log_details} dim={EMBEDDING_DIM}", file=sys.stderr)

# --- Milvus Client ---
MILVUS_ADDRESS = os.getenv("MILVUS_ADDRESS", os.path.join(INDEX_DATA_PATH, "milvus_markdown.db"))
milvus_client = MilvusClient(MILVUS_ADDRESS)


def search(query: str, k: int) -> list[list[SearchResult]]:
    query_vectors = embedding_fn.encode_queries([query])
    res = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=query_vectors,
        limit=k,
        output_fields=list(Entity.model_fields.keys()),
    )
    return res


@mcp.tool(
    name="index_documents",
    description="Index Markdown files for semantic search using Milvus.",
    tags={"index", "vectorize", "store"},
)
async def index_documents(
    current_working_directory: str = Field(description="Current working directory"),
    directory: str = Field("", description="Directory to index"),
    recursive: bool = Field(False, description="Recursively index subdirectories"),
    force_reindex: bool = Field(False, description="Force reindex"),
):
    # TODO: Implement Client Elicitation when it is available in Popular clients
    target_path = os.path.join(current_working_directory, directory)

    if not os.path.exists(target_path):
        return {"message": "Directory does not exist!"}

    if force_reindex:
        if milvus_client.has_collection(COLLECTION_NAME):
            milvus_client.drop_collection(COLLECTION_NAME)
        ensure_collection(milvus_client)

        all_files = list_md_files(target_path, recursive=recursive)
        documents = SimpleDirectoryReader(
            input_files=all_files, required_exts=[".md"]
        ).load_data()
        processed_files = [doc.metadata["file_path"] for doc in documents]

    else:
        changed_files = get_changed_files(target_path, recursive=recursive)

        if not changed_files:
            return {"message": "Already up to date, Nothing to index!"}
        # If not collection exists create a new one
        ensure_collection(milvus_client)
        # Needs to delete the old chunks related to changed files
        for file_path in changed_files:
            try:
                milvus_client.delete(
                    collection_name=COLLECTION_NAME, filter=f"path == '{file_path}'"
                )
            except Exception:
                continue

        # Load only changed files to index
        documents = SimpleDirectoryReader(
            input_files=changed_files, required_exts=[".md"]
        ).load_data()
        # Update tracking file
        processed_files = changed_files

    # Convert to nodes based on markdown structure, then split larger nodes into chunks
    nodes = MarkdownNodeParser().get_nodes_from_documents(documents)
    chunked_nodes = TokenTextSplitter(
        chunk_size=512, chunk_overlap=100
    ).get_nodes_from_documents(nodes)
    chunked_nodes = [node for node in chunked_nodes if node.text.strip()]

    # Extract text from nodes and embed (batch to stay under API limits)
    texts = [node.text for node in chunked_nodes]
    BATCH_SIZE = 100
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        vectors.extend(embedding_fn.encode_documents(batch))
    data = [
        {
            "vector": vector,
            "text": node.text,
            "filename": node.metadata["file_name"],
            "path": node.metadata["file_path"],
        }
        for vector, node in zip(vectors, chunked_nodes)
    ]
    res = milvus_client.insert(collection_name=COLLECTION_NAME, data=data)

    # Update tracking file
    update_tracking_file(processed_files)

    return {
        **res,
        "message": "Full reindex" if force_reindex else "Incremental update",
        "processed_files": len(processed_files),
        "total_chunks": len(chunked_nodes),
        "files": [os.path.basename(f) for f in processed_files],
    }


@mcp.tool(
    name="search_documents",
    description="Search for semantically relevant documents based on query",
    tags={"search", "query"},
)
async def search_documents(
    query: str = Field(description="Query to search for"),
    k: int = Field(5, description="Number of documents to return"),
):
    results = search(query, k=k)

    return "\n---\n".join(
        [
            f"File: **{res.entity.filename}**\n---\nText: {res.entity.text}\n---\n"
            for res in results[0]
        ]
    )  # Iterate through the relevent docs and append the text


@mcp.tool(
    name="clear_index",
    description="Clear the vector database's collection and reset the tracking file",
    tags={"clear", "reset"},
)
async def clear_index():
    ensure_collection(milvus_client)
    res = milvus_client.delete(collection_name=COLLECTION_NAME, filter="id >= 0")
    update_tracking_file([], is_clear=True)
    return res


if __name__ == "__main__":
    mcp.run(transport="stdio")
