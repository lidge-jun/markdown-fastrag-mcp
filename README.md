<div align="center">
  <img src="docs/banner.png" alt="MCP-Markdown-RAG" width="800" style="border-radius:10px;"/>
  <h1>MCP-Markdown-RAG</h1>
  <p>
  <img alt="GitHub forks" src="https://img.shields.io/github/forks/bitkyc08-arch/mcp-markdown-rag"/>
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/bitkyc08-arch/mcp-markdown-rag">
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/bitkyc08-arch/mcp-markdown-rag">
</p>
<p>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License" />
  </a>
  <img src="https://img.shields.io/badge/MCP-Server-blue"/>
</p>
</div>

> **Fork of [Zackriya-Solutions/MCP-Markdown-RAG](https://github.com/Zackriya-Solutions/MCP-Markdown-RAG)**
> with multi-provider embedding support and configurable Milvus address.

A **Model Context Protocol (MCP)** server that provides a **RAG engine** for your markdown documents. This server uses a Milvus vector database to index your notes, enabling Large Language Models (LLMs) to perform semantic search and retrieve relevant content from your files.

## What's Different in This Fork

| Feature | Upstream | This Fork |
|---------|----------|-----------|
| Embedding provider | Local model only | **Gemini, Voyage, OpenAI, Local** (configurable) |
| Milvus address | Local file DB only | **Configurable** via `MILVUS_ADDRESS` env var |
| Embedding dimension | Fixed 768 | **Configurable** via `EMBEDDING_DIM` env var |
| Batch embedding | N/A | **Batched** (100 per request) to stay under API limits |

## Key Features

- **Semantic Search for Markdown**: Find document sections based on conceptual meaning, not just keywords.
- **Multi-Provider Embeddings**: Choose between Gemini, Voyage AI, OpenAI, or the default local model.
- **Incremental Indexing**: Only re-indexes changed files by comparing hashes and timestamps.
- **MCP Compatible**: Integrates with any MCP-supported host (Claude Code, Claude Desktop, Windsurf, Cursor, etc.).

## Installation & Setup

Requires **uv** (Python package manager).

### Step 1: Clone

```bash
git clone https://github.com/bitkyc08-arch/mcp-markdown-rag.git
```

### Step 2: Configure Your Host App

Add to your MCP host configuration:

```json
{
  "mcpServers": {
    "markdown-rag": {
      "command": "uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/mcp-markdown-rag",
        "run",
        "server.py"
      ],
      "env": {
        "EMBEDDING_PROVIDER": "gemini",
        "EMBEDDING_MODEL": "gemini-embedding-001",
        "EMBEDDING_DIM": "768",
        "GEMINI_API_KEY": "${GEMINI_API_KEY}"
      }
    }
  }
}
```

> Replace `/ABSOLUTE/PATH/TO/mcp-markdown-rag` with the actual path.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `local` | Embedding provider: `gemini`, `voyage`, `openai`, or `local` |
| `EMBEDDING_MODEL` | (provider default) | Model name override (e.g. `gemini-embedding-001`, `voyage-3`, `text-embedding-3-small`) |
| `EMBEDDING_DIM` | `768` | Embedding vector dimension |
| `GEMINI_API_KEY` | — | API key for Gemini provider |
| `VOYAGE_API_KEY` | — | API key for Voyage provider |
| `OPENAI_API_KEY` | — | API key for OpenAI provider |
| `MILVUS_ADDRESS` | `.db/milvus_markdown.db` | Milvus server address or local file path |

## Available Tools

- **`index_documents`** — Index markdown files for semantic search.
  - `current_working_directory` (string, required): Base directory path.
  - `directory` (string, optional): Subdirectory to index. Defaults to `""`.
  - `recursive` (boolean, optional): Recursively index subdirectories. Defaults to `false`.
  - `force_reindex` (boolean, optional): Clear and rebuild index. Defaults to `false`.

- **`search_documents`** — Search indexed documents by semantic similarity.
  - `query` (string, required): Natural language query.
  - `k` (integer, optional): Number of results to return. Defaults to `5`.

- **`clear_index`** — Clear the vector database and reset tracking.

## Debugging

```bash
npx @modelcontextprotocol/inspector uv --directory /ABSOLUTE/PATH/TO/mcp-markdown-rag run server.py
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

This is a fork of [Zackriya-Solutions/MCP-Markdown-RAG](https://github.com/Zackriya-Solutions/MCP-Markdown-RAG). See individual source files for modification notices as required by the Apache 2.0 license.
