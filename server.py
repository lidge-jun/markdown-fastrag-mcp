# Modified from original MCP-Markdown-RAG by Zackriya Solutions
# (https://github.com/Zackriya-Solutions/MCP-Markdown-RAG)
# Changes: multi-provider embedding support (Gemini/Voyage/OpenAI),
#          configurable Milvus address, batch embedding for API limits.

import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

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
    get_index_delta,
    list_md_files,
    update_tracking_file,
)

mcp = FastMCP(
    "markdown-fastrag-mcp",
    instructions="""This MCP server provides semantic search capabilities over markdown files using 
    vector embeddings and Milvus database. It enables you to index markdown documents and perform 
    intelligent searches to find relevant content based on semantic similarity rather than just 
    keyword matching.""",
)


if not os.path.exists(INDEX_DATA_PATH):
    os.makedirs(INDEX_DATA_PATH)

# --- Workspace Lock ---
# When set, all index_documents calls use this as the root directory,
# ignoring the current_working_directory/directory params from the agent.
MARKDOWN_WORKSPACE = os.getenv("MARKDOWN_WORKSPACE", "").strip() or None
if MARKDOWN_WORKSPACE:
    print(f"[Workspace] Locked to: {MARKDOWN_WORKSPACE}", file=sys.stderr)


# --- Embedding Provider Selection ---
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
embedding_log_details = ""


def _get_int_env(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            value = default
    if minimum is not None and value < minimum:
        return minimum
    return value


MARKDOWN_CHUNK_SIZE = _get_int_env("MARKDOWN_CHUNK_SIZE", 2048, minimum=32)
MARKDOWN_CHUNK_OVERLAP = _get_int_env("MARKDOWN_CHUNK_OVERLAP", 100, minimum=0)
EMBEDDING_BATCH_SIZE = _get_int_env("EMBEDDING_BATCH_SIZE", 250, minimum=1)
EMBEDDING_BATCH_DELAY_MS = _get_int_env("EMBEDDING_BATCH_DELAY_MS", 0, minimum=0)
EMBEDDING_CONCURRENT_BATCHES = _get_int_env("EMBEDDING_CONCURRENT_BATCHES", 4, minimum=1)
MARKDOWN_BG_MAX_JOBS = _get_int_env("MARKDOWN_BG_MAX_JOBS", 1, minimum=1)
MARKDOWN_BG_JOB_TTL_SECONDS = _get_int_env("MARKDOWN_BG_JOB_TTL_SECONDS", 1800, minimum=1)
MIN_CHUNK_TOKENS = _get_int_env("MIN_CHUNK_TOKENS", 300, minimum=0)
DEDUP_MAX_PER_FILE = _get_int_env("DEDUP_MAX_PER_FILE", 1, minimum=0)


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
print(
    "[Indexing] "
    f"chunk_size={MARKDOWN_CHUNK_SIZE} "
    f"chunk_overlap={MARKDOWN_CHUNK_OVERLAP} "
    f"batch_size={EMBEDDING_BATCH_SIZE} "
    f"batch_delay_ms={EMBEDDING_BATCH_DELAY_MS} "
    f"min_chunk_tokens={MIN_CHUNK_TOKENS}",
    file=sys.stderr,
)
print(
    "[Background] "
    f"max_jobs={MARKDOWN_BG_MAX_JOBS} "
    f"job_ttl_seconds={MARKDOWN_BG_JOB_TTL_SECONDS}",
    file=sys.stderr,
)

# --- Milvus Client ---
MILVUS_ADDRESS = os.getenv("MILVUS_ADDRESS", os.path.join(INDEX_DATA_PATH, "milvus_markdown.db"))
milvus_client = MilvusClient(MILVUS_ADDRESS)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clamp_percent(value: int) -> int:
    return max(0, min(100, int(value)))


def _safe_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _normalize_recursive(recursive: bool) -> tuple[bool, str | None]:
    if recursive:
        return True, None
    warning = "recursive=false is not supported; forcing recursive=true."
    print(f"[Indexing] {warning}", file=sys.stderr, flush=True)
    return True, warning


@dataclass
class JobRecord:
    job_id: str
    target_path: str
    status: str = "queued"  # queued | running | succeeded | failed
    params: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime = field(default_factory=_utc_now)
    progress_percent: int = 0
    phase: str = "pending"  # pending | scan | chunk | embed | insert | done | failed
    processed_files: int = 0
    total_files: int = 0
    processed_chunks: int = 0
    total_chunks: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None
    note: str | None = None

    def touch(self) -> None:
        self.updated_at = _utc_now()


_jobs: dict[str, JobRecord] = {}
_job_tasks: dict[str, asyncio.Task] = {}
_active_targets: set[str] = set()
_jobs_lock = asyncio.Lock()
_tracking_lock = asyncio.Lock()


def _job_elapsed_seconds(job: JobRecord) -> float:
    if not job.started_at:
        return 0.0
    end = job.finished_at or _utc_now()
    return round(max(0.0, (end - job.started_at).total_seconds()), 2)


def _serialize_job(job: JobRecord) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "phase": job.phase,
        "progress_percent": job.progress_percent,
        "processed_files": job.processed_files,
        "total_files": job.total_files,
        "processed_chunks": job.processed_chunks,
        "total_chunks": job.total_chunks,
        "target_path": job.target_path,
        "params": job.params,
        "note": job.note,
        "result": job.result,
        "error": job.error,
        "created_at": _safe_iso(job.created_at),
        "started_at": _safe_iso(job.started_at),
        "finished_at": _safe_iso(job.finished_at),
        "updated_at": _safe_iso(job.updated_at),
        "elapsed_seconds": _job_elapsed_seconds(job),
    }


def _set_job_progress(
    job: JobRecord,
    *,
    phase: str,
    percent: int,
    message: str | None = None,
    processed_files: int | None = None,
    total_files: int | None = None,
    processed_chunks: int | None = None,
    total_chunks: int | None = None,
) -> None:
    job.phase = phase
    job.progress_percent = _clamp_percent(percent)
    if message is not None:
        job.note = message
    if processed_files is not None:
        job.processed_files = max(0, processed_files)
    if total_files is not None:
        job.total_files = max(0, total_files)
    if processed_chunks is not None:
        job.processed_chunks = max(0, processed_chunks)
    if total_chunks is not None:
        job.total_chunks = max(0, total_chunks)
    job.touch()


def _cleanup_expired_jobs_locked() -> None:
    now = _utc_now()
    expired_job_ids: list[str] = []

    for job_id, job in _jobs.items():
        if job.status not in {"succeeded", "failed"}:
            continue
        if not job.finished_at:
            continue
        if (now - job.finished_at).total_seconds() <= MARKDOWN_BG_JOB_TTL_SECONDS:
            continue
        task = _job_tasks.get(job_id)
        if task is not None and not task.done():
            continue
        expired_job_ids.append(job_id)

    for job_id in expired_job_ids:
        _jobs.pop(job_id, None)
        _job_tasks.pop(job_id, None)


def _active_jobs_locked() -> list[JobRecord]:
    active: list[JobRecord] = []
    for job_id, job in _jobs.items():
        task = _job_tasks.get(job_id)
        if job.status in {"queued", "running"} and (task is None or not task.done()):
            active.append(job)
    active.sort(key=lambda item: item.created_at)
    return active


def _latest_job_locked() -> JobRecord | None:
    if not _jobs:
        return None
    return max(_jobs.values(), key=lambda job: job.created_at)


async def _run_index_job(
    target_path: str,
    recursive: bool,
    force_reindex: bool,
    job: JobRecord | None = None,
) -> dict[str, Any]:
    if not os.path.exists(target_path):
        return {"message": "Directory does not exist!"}

    if job is not None:
        _set_job_progress(job, phase="scan", percent=5, message="Scanning markdown files")

    if force_reindex:
        if milvus_client.has_collection(COLLECTION_NAME):
            milvus_client.drop_collection(COLLECTION_NAME)
        ensure_collection(milvus_client)

        all_files = list_md_files(target_path, recursive=recursive)
        documents = SimpleDirectoryReader(
            input_files=all_files, required_exts=[".md"]
        ).load_data()
        processed_files = [doc.metadata["file_path"] for doc in documents]
        pruned_count = 0
    else:
        # Protect tracking file read/write operations from concurrent jobs.
        async with _tracking_lock:
            changed_files, deleted_files = get_index_delta(target_path, recursive=recursive)

        ensure_collection(milvus_client)
        pruned_count = 0
        for file_path in deleted_files:
            try:
                milvus_client.delete(
                    collection_name=COLLECTION_NAME, filter=f"path == '{file_path}'"
                )
                pruned_count += 1
            except Exception:
                continue
        if pruned_count > 0:
            print(
                f"[Indexer] Pruned {pruned_count} deleted/moved files from index",
                file=sys.stderr,
                flush=True,
            )

        if not changed_files:
            message = (
                f"Pruned {pruned_count} deleted/moved files. No new files to index."
                if pruned_count > 0
                else "Already up to date, Nothing to index!"
            )
            if job is not None:
                _set_job_progress(
                    job,
                    phase="done",
                    percent=100,
                    message=message,
                    processed_files=0,
                    total_files=0,
                    processed_chunks=0,
                    total_chunks=0,
                )
            return {"message": message, "pruned_files": pruned_count}

        # Remove old chunks for changed files before inserting fresh embeddings.
        for file_path in changed_files:
            try:
                milvus_client.delete(
                    collection_name=COLLECTION_NAME, filter=f"path == '{file_path}'"
                )
            except Exception:
                continue

        documents = SimpleDirectoryReader(
            input_files=changed_files, required_exts=[".md"]
        ).load_data()
        processed_files = changed_files

    if job is not None:
        _set_job_progress(
            job,
            phase="chunk",
            percent=20,
            message=f"Chunking {len(processed_files)} files",
            total_files=len(processed_files),
        )

    # Pre-process: strip YAML frontmatter from documents BEFORE chunking.
    # LlamaIndex propagates doc.metadata to all child nodes automatically.
    from chunking import (
        _normalize_meta,
        inject_header_prefix,
        merge_small_chunks,
        strip_frontmatter,
    )

    for doc in documents:
        clean_text, fm = strip_frontmatter(doc.text)
        doc.text = clean_text
        doc.metadata["tags"] = _normalize_meta(fm.get("tags"))
        doc.metadata["aliases"] = _normalize_meta(fm.get("aliases"))

    # Convert to nodes based on markdown structure, then split larger nodes into chunks.
    nodes = MarkdownNodeParser(chunk_size=MARKDOWN_CHUNK_SIZE).get_nodes_from_documents(documents)
    chunk_overlap = min(MARKDOWN_CHUNK_OVERLAP, max(0, MARKDOWN_CHUNK_SIZE - 1))
    chunked_nodes = TokenTextSplitter(
        chunk_size=MARKDOWN_CHUNK_SIZE, chunk_overlap=chunk_overlap
    ).get_nodes_from_documents(nodes)
    chunked_nodes = [node for node in chunked_nodes if node.text.strip()]

    # Post-process: merge small chunks + inject parent header context.
    if MIN_CHUNK_TOKENS > 0:
        chunked_nodes = merge_small_chunks(
            chunked_nodes, MIN_CHUNK_TOKENS, MARKDOWN_CHUNK_SIZE
        )
    chunked_nodes = inject_header_prefix(chunked_nodes)

    # Extract text from nodes and embed (parallel batches for speed).
    texts = [node.text for node in chunked_nodes]
    batches = [texts[i: i + EMBEDDING_BATCH_SIZE] for i in range(0, len(texts), EMBEDDING_BATCH_SIZE)]
    total_batches = len(batches)
    print(
        f"[Embedding] {len(texts)} chunks in {total_batches} batches "
        f"(size={EMBEDDING_BATCH_SIZE}, concurrent={EMBEDDING_CONCURRENT_BATCHES})",
        file=sys.stderr,
        flush=True,
    )

    if job is not None:
        job.total_chunks = len(texts)
        job.touch()

    vectors: list[list[float]] = [None] * len(texts)  # type: ignore[assignment]
    executor = ThreadPoolExecutor(max_workers=EMBEDDING_CONCURRENT_BATCHES)
    loop = asyncio.get_running_loop()

    async def embed_one(batch_idx: int, batch: list[str], offset: int) -> None:
        print(
            f"[Embedding] batch {batch_idx + 1}/{total_batches} "
            f"({offset}~{offset + len(batch)}/{len(texts)})",
            file=sys.stderr,
            flush=True,
        )
        result = await loop.run_in_executor(executor, embedding_fn.encode_documents, batch)
        for j, vec in enumerate(result):
            vectors[offset + j] = vec

    # Process in waves of EMBEDDING_CONCURRENT_BATCHES.
    for wave_start in range(0, total_batches, EMBEDDING_CONCURRENT_BATCHES):
        wave = []
        wave_end = min(wave_start + EMBEDDING_CONCURRENT_BATCHES, total_batches)
        for k in range(wave_start, wave_end):
            offset = k * EMBEDDING_BATCH_SIZE
            wave.append(embed_one(k, batches[k], offset))
        await asyncio.gather(*wave)

        if job is not None and total_batches > 0:
            completed_batches = wave_end
            processed_chunks = min(completed_batches * EMBEDDING_BATCH_SIZE, len(texts))
            embed_percent = 20 + int((completed_batches / total_batches) * 60)
            _set_job_progress(
                job,
                phase="embed",
                percent=embed_percent,
                message=f"Embedding batch {completed_batches}/{total_batches}",
                processed_chunks=processed_chunks,
                total_chunks=len(texts),
            )

        if EMBEDDING_BATCH_DELAY_MS > 0:
            await asyncio.sleep(EMBEDDING_BATCH_DELAY_MS / 1000.0)

    executor.shutdown(wait=False)
    data = [
        {
            "vector": vector,
            "text": node.text,
            "filename": node.metadata["file_name"],
            "path": node.metadata["file_path"],
            "tags": node.metadata.get("tags", ""),
            "aliases": node.metadata.get("aliases", ""),
        }
        for vector, node in zip(vectors, chunked_nodes)
    ]
    milvus_insert_batch = _get_int_env("MILVUS_INSERT_BATCH", 5000, minimum=1)
    total_insert_batches = max(1, (len(data) + milvus_insert_batch - 1) // milvus_insert_batch)
    res: dict[str, Any] = {}

    for insert_idx, i in enumerate(range(0, len(data), milvus_insert_batch), start=1):
        batch = data[i: i + milvus_insert_batch]
        res = milvus_client.insert(collection_name=COLLECTION_NAME, data=batch)
        if job is not None:
            insert_percent = 80 + int((insert_idx / total_insert_batches) * 15)
            _set_job_progress(
                job,
                phase="insert",
                percent=insert_percent,
                message=f"Inserting batch {insert_idx}/{total_insert_batches}",
                processed_chunks=len(data),
                total_chunks=len(texts),
            )

    # Protect tracking file write from concurrent jobs.
    async with _tracking_lock:
        update_tracking_file(processed_files)

    result = {
        **res,
        "message": "Full reindex" if force_reindex else "Incremental update",
        "processed_files": len(processed_files),
        "total_chunks": len(chunked_nodes),
        "files": [os.path.basename(f) for f in processed_files],
        "pruned_files": pruned_count,
    }

    if job is not None:
        _set_job_progress(
            job,
            phase="done",
            percent=100,
            message="Index job completed",
            processed_files=len(processed_files),
            total_files=len(processed_files),
            processed_chunks=len(chunked_nodes),
            total_chunks=len(chunked_nodes),
        )

    return result


async def _execute_index_job(
    job_id: str,
    target_path: str,
    recursive: bool,
    force_reindex: bool,
) -> None:
    async with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.status = "running"
        job.started_at = _utc_now()
        job.touch()
        _set_job_progress(job, phase="scan", percent=3, message="Background job started")

    try:
        result = await _run_index_job(
            target_path=target_path,
            recursive=recursive,
            force_reindex=force_reindex,
            job=job,
        )
        job.result = result
        if job.status != "failed":
            job.status = "succeeded"
        if job.progress_percent < 100:
            _set_job_progress(job, phase="done", percent=100, message="Index job completed")
    except Exception as exc:
        job.status = "failed"
        job.phase = "failed"
        job.error = str(exc)
        job.touch()
        print(f"[Background] Job {job_id} failed: {exc}", file=sys.stderr, flush=True)
    finally:
        async with _jobs_lock:
            finished = _jobs.get(job_id)
            if finished is not None:
                finished.finished_at = _utc_now()
                finished.touch()
            _active_targets.discard(target_path)
            _cleanup_expired_jobs_locked()

def search(query: str, k: int, scope_path: str = "") -> list[list[SearchResult]]:
    # Oversampling: fetch extra candidates so dedup can fill k slots with diverse files.
    oversample = max(k * 5, 20) if DEDUP_MAX_PER_FILE > 0 else k
    query_vectors = embedding_fn.encode_queries([query])
    search_params = dict(
        collection_name=COLLECTION_NAME,
        data=query_vectors,
        limit=oversample,
        output_fields=list(Entity.model_fields.keys()),
    )
    if scope_path:
        scope_prefix = os.path.normpath(scope_path)
        if not scope_prefix.endswith(os.sep):
            scope_prefix += os.sep
        safe_prefix = scope_prefix.replace("'", "\\'")
        search_params["filter"] = f"path like '{safe_prefix}%'"
    res = milvus_client.search(**search_params)

    if not res or DEDUP_MAX_PER_FILE <= 0:
        return res

    # Dedup: limit results per file path to ensure diverse sources.
    deduped = []
    file_count: dict[str, int] = {}
    for hit in res[0]:
        fp = hit.entity.path
        count = file_count.get(fp, 0)
        if count < DEDUP_MAX_PER_FILE:
            deduped.append(hit)
            file_count[fp] = count + 1
        if len(deduped) >= k:
            break
    return [deduped]


@mcp.tool(
    name="index_documents",
    description=(
        "Index Markdown files. Returns IMMEDIATELY with job_id (non-blocking). "
        "Poll get_index_status until status='succeeded' before calling search_documents. "
        "Do NOT search while indexing. "
        "WARNING: If MARKDOWN_WORKSPACE env is set, the workspace root is locked and "
        "current_working_directory/directory params are ignored. "
        "If not set, always pass the vault root as current_working_directory to avoid "
        "partial tracking file corruption."
    ),
    tags={"index", "background", "async"},
)
async def index_documents(
    current_working_directory: str = Field(description="Current working directory"),
    directory: str = Field("", description="Directory to index"),
    recursive: bool = Field(True, description="Recursively index subdirectories"),
    force_reindex: bool = Field(False, description="Force reindex"),
):
    # If MARKDOWN_WORKSPACE is set, override agent-supplied paths.
    if MARKDOWN_WORKSPACE:
        target_path = MARKDOWN_WORKSPACE
    else:
        target_path = os.path.join(current_working_directory, directory)
    recursive, recursive_warning = _normalize_recursive(recursive)

    # Block force_reindex via MCP — shell only
    if force_reindex:
        return {
            "accepted": False,
            "status": "rejected",
            "message": (
                "force_reindex=true is not allowed via MCP. "
                "Use shell: uv run python reindex.py /path --force"
            ),
        }

    if not os.path.exists(target_path):
        return {
            "accepted": False,
            "status": "rejected",
            "message": "Directory does not exist!",
            "job_id": None,
        }

    async with _jobs_lock:
        _cleanup_expired_jobs_locked()
        active_jobs = _active_jobs_locked()

        if force_reindex and active_jobs:
            current = _serialize_job(active_jobs[0])
            return {
                "accepted": False,
                "status": "rejected",
                "message": "Cannot force reindex while another index job is active.",
                "job_id": None,
                "current_job": current,
            }

        if len(active_jobs) >= MARKDOWN_BG_MAX_JOBS:
            current = _serialize_job(active_jobs[0])
            return {
                "accepted": False,
                "status": "rejected",
                "message": (
                    f"Maximum active index jobs reached "
                    f"({len(active_jobs)}/{MARKDOWN_BG_MAX_JOBS})."
                ),
                "job_id": None,
                "current_job": current,
            }

        if target_path in _active_targets:
            same_target = next((job for job in active_jobs if job.target_path == target_path), None)
            current = _serialize_job(same_target) if same_target else None
            return {
                "accepted": False,
                "status": "rejected",
                "message": "An index job for the same target path is already active.",
                "job_id": None,
                "current_job": current,
            }

        job_id = uuid4().hex[:8]
        params = {
            "current_working_directory": current_working_directory,
            "directory": directory,
            "recursive": recursive,
            "force_reindex": force_reindex,
        }
        job = JobRecord(job_id=job_id, target_path=target_path, params=params, note=recursive_warning)
        _jobs[job_id] = job
        _active_targets.add(target_path)

        task = asyncio.create_task(
            _execute_index_job(
                job_id=job_id,
                target_path=target_path,
                recursive=recursive,
                force_reindex=force_reindex,
            )
        )
        _job_tasks[job_id] = task

        response: dict[str, Any] = {
            "accepted": True,
            "status": "queued",
            "message": "Index job accepted and queued.",
            "job_id": job_id,
            "started_at": _safe_iso(job.started_at),
            "created_at": _safe_iso(job.created_at),
        }
        if recursive_warning:
            response["warning"] = recursive_warning
        return response


@mcp.tool(
    name="get_index_status",
    description="Get current status for a background markdown indexing job.",
    tags={"index", "background", "status"},
)
async def get_index_status(
    job_id: str = Field("", description="Background job ID; empty returns latest job."),
):
    lookup_id = job_id.strip()

    async with _jobs_lock:
        _cleanup_expired_jobs_locked()
        active_jobs = _active_jobs_locked()

        job = _jobs.get(lookup_id) if lookup_id else _latest_job_locked()
        if job is None:
            return {
                "found": False,
                "message": "No indexing jobs found.",
                "job_id": None,
                "active_jobs": 0,
            }

        payload = _serialize_job(job)
        payload["found"] = True
        payload["active_jobs"] = len(active_jobs)
        return payload


@mcp.tool(
    name="search_documents",
    description=(
        "Search for semantically relevant documents based on query. "
        "Use scope_path to limit results to a specific subdirectory."
    ),
    tags={"search", "query"},
)
async def search_documents(
    query: str = Field(description="Query to search for"),
    k: int = Field(5, description="Number of documents to return"),
    scope_path: str = Field(
        "",
        description=(
            "Limit search to files under this absolute path prefix. "
            "Empty string searches all indexed documents."
        ),
    ),
):
    # Validate scope_path against workspace
    if scope_path and MARKDOWN_WORKSPACE:
        workspace = os.path.realpath(MARKDOWN_WORKSPACE)
        scope = os.path.realpath(scope_path)
        try:
            if os.path.commonpath([workspace, scope]) != workspace:
                return (
                    f"Error: scope_path must be under MARKDOWN_WORKSPACE "
                    f"({MARKDOWN_WORKSPACE}). Got: {scope_path}"
                )
        except ValueError:
            return (
                f"Error: scope_path must be under MARKDOWN_WORKSPACE "
                f"({MARKDOWN_WORKSPACE}). Got: {scope_path}"
            )

    results = search(query, k=k, scope_path=scope_path)

    parts = []
    for res in results[0]:
        entry = (
            f"File: **{res.entity.filename}** (relevance: {res.distance:.1%})\n"
            f"Path: `{res.entity.path}`\n"
        )
        if res.entity.tags:
            entry += f"Tags: {res.entity.tags}\n"
        if res.entity.aliases:
            entry += f"Aliases: {res.entity.aliases}\n"
        entry += f"---\nText: {res.entity.text}\n---\n"
        parts.append(entry)

    return "\n---\n".join(parts)


@mcp.tool(
    name="clear_index",
    description=(
        "DESTRUCTIVE: Clear the entire vector database collection and reset the tracking file. "
        "This deletes ALL indexed documents and cannot be undone. "
        "A full reindex is required after clearing. Only use when explicitly requested by the user."
    ),
    tags={"clear", "reset"},
)
async def clear_index():
    async with _jobs_lock:
        _cleanup_expired_jobs_locked()
        active_jobs = _active_jobs_locked()
        if active_jobs:
            return {
                "message": "Cannot clear index while background indexing is active.",
                "active_jobs": len(active_jobs),
                "current_job": _serialize_job(active_jobs[0]),
            }

    ensure_collection(milvus_client)
    res = milvus_client.delete(collection_name=COLLECTION_NAME, filter="id >= 0")
    async with _tracking_lock:
        update_tracking_file([], is_clear=True)
    return res


if __name__ == "__main__":
    mcp.run(transport="stdio")
