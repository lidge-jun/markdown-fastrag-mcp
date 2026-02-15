# MCP-Markdown-RAG

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![MCP Server](https://img.shields.io/badge/MCP-Server-blue)](https://modelcontextprotocol.io)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.10-blue.svg)](https://python.org/)

A semantic search engine for your markdown documents. An MCP server that indexes notes, docs, and knowledge bases into a Milvus vector database, letting AI assistants find relevant content by **meaning**.

> Ask *"what are the tradeoffs of microservices?"* and find your notes about service boundaries, distributed systems, and API design — even if none of them mention "microservices."

## Features

- **Semantic matching** — finds conceptually related content, not just keyword hits
- **Multi-provider embeddings** — Gemini, OpenAI, Vertex AI, Voyage, or local models
- **Smart incremental indexing** — mtime/size fast-path skips unchanged files without reading them; hash only computed when metadata changes
- **Single-pass delta scan** — detects new, changed, and deleted files in one directory walk
- **Stale vector pruning** — automatically removes vectors for deleted or moved files from Milvus
- **Batch embedding** — concurrent batches with rate-limit retry (429 exponential backoff)
- **Batch insert** — chunked Milvus inserts to stay under the gRPC 64MB message limit
- **Shell reindex CLI** — `reindex.py` for large-scale indexing with real-time progress logs
- **Configurable exclusions** — skip directories (`node_modules`, `.git`, `_legacy`) and files (`AGENTS.md`) via env
- **Milvus Standalone support** — connect to a Docker-based Milvus server for multi-agent concurrent access
- **MCP native** — works with any MCP host (Claude Code, Cursor, Windsurf, VS Code, Antigravity, Codex, etc.)

## Architecture

```mermaid
graph TB
    subgraph MCP["MCP Server (server.py)"]
        direction TB
        IDX["index_documents<br/>Incremental Indexing"]
        SEARCH["search_documents<br/>Semantic Search"]
        CLEAR["clear_index<br/>Reset"]
    end

    subgraph Indexing["Indexing Engine (utils.py)"]
        DELTA["get_index_delta<br/>Single-pass Delta Scan"]
        TRACK["index_tracking.json<br/>mtime / size / hash"]
        CHUNK["llama-index<br/>SentenceSplitter"]
    end

    subgraph Embed["Embedding Providers"]
        VERTEX["Vertex AI<br/>gemini-embedding-001"]
        GEMINI["Gemini API<br/>OpenAI-compat"]
        OAI["OpenAI / Compatible"]
        VOYAGE["Voyage AI<br/>voyage-3"]
        LOCAL["Milvus Built-in<br/>DefaultEmbeddingFunction"]
    end

    subgraph Store["Vector Store"]
        MILVUS["Milvus Standalone<br/>Docker (gRPC)"]
        LITE["Milvus Lite<br/>SQLite (local)"]
    end

    IDX --> DELTA --> CHUNK --> Embed --> Store
    SEARCH --> Embed --> Store
    DELTA <--> TRACK

    style MCP fill:#2d3748,color:#e2e8f0
    style Embed fill:#553c9a,color:#e9d8fd
    style Store fill:#2a4365,color:#bee3f8
    style Indexing fill:#22543d,color:#c6f6d5
```

## How It Works

```mermaid
flowchart LR
    A["📁 Markdown Files"] -->|"directory walk\n+ exclude filter"| B["🔍 Delta Scan\nmtime/size check"]
    B -->|changed| C["✂️ Chunk\nSentenceSplitter"]
    B -->|unchanged| SKIP["⏭️ Skip"]
    B -->|deleted| PRUNE["🗑️ Prune\nMilvus delete"]
    C --> D["🧠 Embed\nVertex/Gemini/OpenAI"]
    D -->|"batch insert"| E["💾 Milvus\nVector Store"]

    F["🔎 Search Query"] --> D
    D -->|"cosine similarity"| G["📊 Top-K Results\nwith relevance %"]

    style A fill:#2d3748,color:#e2e8f0
    style D fill:#553c9a,color:#e9d8fd
    style E fill:#2a4365,color:#bee3f8
    style G fill:#22543d,color:#c6f6d5
    style PRUNE fill:#742a2a,color:#fed7d7
```

## Quick Start

Requires [uv](https://docs.astral.sh/uv/) (Python package manager).

### 1. Clone

```bash
git clone https://github.com/bitkyc08-arch/mcp-markdown-rag.git
```

### 2. Configure

Add to your MCP host config:

```json
{
  "mcpServers": {
    "markdown-rag": {
      "command": "uv",
      "args": [
        "--directory", "/path/to/mcp-markdown-rag",
        "run", "server.py"
      ],
      "env": {
        "EMBEDDING_PROVIDER": "gemini",
        "EMBEDDING_MODEL": "gemini-embedding-001",
        "EMBEDDING_DIM": "768",
        "GEMINI_API_KEY": "${GEMINI_API_KEY}",
        "MILVUS_ADDRESS": "http://localhost:19530"
      }
    }
  }
}
```

> **Tip**: For local-only use (no Docker), omit `MILVUS_ADDRESS` — it defaults to a local SQLite-based Milvus Lite file (`.db/milvus_markdown.db`).

## Embedding Providers

| Provider              | `EMBEDDING_PROVIDER` | Default Model            | Auth            |
| --------------------- | -------------------- | ------------------------ | --------------- |
| **Vertex AI**         | `vertex`             | `gemini-embedding-001`   | Service Account |
| **Gemini**            | `gemini`             | `gemini-embedding-001`   | API key         |
| **OpenAI**            | `openai`             | `text-embedding-3-small` | API key         |
| **OpenAI-compatible** | `openai-compatible`  | `text-embedding-3-small` | API key         |
| **Voyage**            | `voyage`             | `voyage-3`               | API key         |
| **Local**             | `local`              | Milvus built-in (768d)   | —               |

<details>
<summary><strong>Vertex AI</strong> — Google Cloud 프로덕션 권장</summary>

Google Cloud의 Vertex AI를 통해 `gemini-embedding-001` 모델을 사용합니다. API key 대신 **Service Account 인증**을 사용하며, OAuth 토큰이 자동 갱신됩니다. 프로덕션 환경에서 가장 안정적입니다.

**장점**: 높은 Rate Limit, 자동 토큰 갱신, GCP 프로젝트 단위 빌링
**단점**: GCP 프로젝트 + Service Account 설정 필요

**사전 준비**:
1. GCP 프로젝트 생성 & Vertex AI API 활성화
2. Service Account 생성 → JSON 키 다운로드
3. `Vertex AI User` 역할 부여

```json
{
  "EMBEDDING_PROVIDER": "vertex",
  "EMBEDDING_MODEL": "gemini-embedding-001",
  "EMBEDDING_DIM": "768",
  "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/service-account.json",
  "VERTEX_PROJECT": "your-gcp-project-id",
  "VERTEX_LOCATION": "us-central1"
}
```

**참고**: `VERTEX_LOCATION`은 모델 사용 가능 리전에 맞춰야 합니다. `gemini-embedding-001`은 `us-central1`에서 사용 가능. 전체 리전 목록은 [Vertex AI 문서](https://cloud.google.com/vertex-ai/docs/general/locations)를 참고.

</details>

<details>
<summary><strong>Gemini</strong> — 빠른 시작에 가장 쉬움</summary>

Google AI Studio의 Gemini API를 사용합니다. API key 하나면 바로 사용 가능해서 가장 간단합니다. 내부적으로 OpenAI-compatible 엔드포인트(`generativelanguage.googleapis.com/v1beta/openai/`)를 사용합니다.

**장점**: 가입 후 즉시 사용, 무료 Tier 있음
**단점**: Rate Limit이 Vertex 대비 낮음 (분당 1,500 RPM 기본)

**사전 준비**:
1. [Google AI Studio](https://aistudio.google.com/)에서 API key 발급

```json
{
  "EMBEDDING_PROVIDER": "gemini",
  "EMBEDDING_MODEL": "gemini-embedding-001",
  "EMBEDDING_DIM": "768",
  "GEMINI_API_KEY": "your-api-key"
}
```

**참고**: 대량 인덱싱(1000+ 파일) 시 429 에러가 발생할 수 있습니다. `EMBEDDING_BATCH_DELAY_MS=1000`으로 설정하면 안정적입니다.

</details>

<details>
<summary><strong>OpenAI</strong> — text-embedding-3 시리즈</summary>

OpenAI의 임베딩 API를 사용합니다. `text-embedding-3-small` (1536d)과 `text-embedding-3-large` (3072d) 모델을 지원합니다. `EMBEDDING_DIM`으로 차원을 줄일 수 있습니다 (Matryoshka representation).

**장점**: 높은 품질, 차원 축소 지원
**단점**: 유료 (small: $0.02/1M tokens, large: $0.13/1M tokens)

**사전 준비**:
1. [OpenAI Platform](https://platform.openai.com/)에서 API key 발급

```json
{
  "EMBEDDING_PROVIDER": "openai",
  "EMBEDDING_MODEL": "text-embedding-3-small",
  "EMBEDDING_DIM": "768",
  "OPENAI_API_KEY": "sk-..."
}
```

**참고**: `EMBEDDING_DIM`을 768로 설정하면 원래 1536d 벡터를 768d로 줄여서 저장합니다. 검색 품질은 소폭 감소하지만 스토리지와 속도가 개선됩니다.

</details>

<details>
<summary><strong>OpenAI-compatible</strong> — 자체 호스팅 / 써드파티 API</summary>

OpenAI API 형식을 따르는 모든 임베딩 서비스에 연결합니다. Ollama, LM Studio, Azure OpenAI, Together AI, Fireworks AI 등 다양한 서비스와 호환됩니다.

**장점**: 자체 호스팅 모델 사용 가능, 프라이버시 보장
**단점**: 서비스별 설정이 다를 수 있음

```json
{
  "EMBEDDING_PROVIDER": "openai-compatible",
  "EMBEDDING_MODEL": "nomic-embed-text",
  "EMBEDDING_DIM": "768",
  "EMBEDDING_API_KEY": "your-api-key-or-dummy",
  "EMBEDDING_BASE_URL": "http://localhost:11434/v1"
}
```

**Ollama 예시**: Ollama에서 `nomic-embed-text`를 사용하려면:

```bash
ollama pull nomic-embed-text
# EMBEDDING_BASE_URL=http://localhost:11434/v1
# EMBEDDING_API_KEY=ollama  (아무 값이나 OK)
```

**Azure OpenAI 예시**:

```json
{
  "EMBEDDING_BASE_URL": "https://your-resource.openai.azure.com/openai/deployments/your-deployment",
  "EMBEDDING_API_KEY": "your-azure-api-key"
}
```

</details>

<details>
<summary><strong>Voyage</strong> — Retrieval 특화 임베딩</summary>

Voyage AI의 임베딩 모델을 사용합니다. `voyage-3`은 검색(retrieval) 태스크에 최적화되어 있어서 RAG에 특히 적합합니다. Anthropic이 Claude에 사용하는 임베딩 provider로도 알려져 있습니다.

**장점**: RAG/검색 품질 최상위권, 긴 컨텍스트 지원 (최대 32K tokens)
**단점**: 유료 ($0.06/1M tokens), 무료 Tier 제한적

**사전 준비**:
1. [Voyage AI](https://www.voyageai.com/)에서 API key 발급

```json
{
  "EMBEDDING_PROVIDER": "voyage",
  "EMBEDDING_MODEL": "voyage-3",
  "VOYAGE_API_KEY": "pa-..."
}
```

**사용 가능 모델**:

| 모델            | 차원 | 최대 토큰 | 용도        |
| --------------- | ---- | --------- | ----------- |
| `voyage-3`      | 1024 | 32K       | 범용 (권장) |
| `voyage-3-lite` | 512  | 32K       | 경량/저비용 |
| `voyage-code-3` | 1024 | 32K       | 코드 특화   |

**참고**: `EMBEDDING_DIM`을 별도 설정하지 않아도 됩니다. Voyage는 모델별 고정 차원을 사용합니다.

</details>

<details>
<summary><strong>Local</strong> — 오프라인 / 무료</summary>

Milvus에 내장된 기본 임베딩 함수를 사용합니다 (`DefaultEmbeddingFunction`, 768d). 인터넷 연결이나 API key 없이 완전한 로컬 환경에서 동작합니다.

**장점**: 무료, 오프라인 사용, API 의존성 없음
**단점**: 클라우드 모델 대비 검색 품질 낮음, 첫 실행 시 모델 다운로드에 시간 소요

```json
{
  "EMBEDDING_PROVIDER": "local"
}
```

별도 환경변수 설정이 필요 없습니다. `EMBEDDING_PROVIDER`를 생략해도 기본값이 `local`입니다. 테스트나 프로토타이핑에 적합합니다.

</details>

## Tools

| Tool               | Description                                                                                                                             |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| `index_documents`  | Index markdown files with incremental updates. Automatically detects new, changed, and deleted files. Prunes stale vectors from Milvus. |
| `search_documents` | Semantic search across indexed documents. Returns top-k results with relevance scores and file paths.                                   |
| `clear_index`      | Reset the vector database and tracking state.                                                                                           |

## Incremental Indexing & Pruning

The indexing engine uses a **single-pass delta scan** to efficiently detect what changed:

```mermaid
flowchart TD
    START["Directory Walk"] --> NEW{"New file?"}
    NEW -->|yes| INDEX["✅ Index"]
    NEW -->|no| META{"mtime/size same?"}
    META -->|yes| SKIP["⏭️ Skip\n(no file read)"]
    META -->|no| HASH{"Compute hash\nContent changed?"}
    HASH -->|yes| REINDEX["🔄 Re-index"]
    HASH -->|no| UPDATE["📝 Update tracking\n(metadata only)"]
    START --> MISSING{"Tracked but\nmissing from disk?"}
    MISSING -->|yes| PRUNE["🗑️ Prune from Milvus\n+ remove tracking"]

    style INDEX fill:#22543d,color:#c6f6d5
    style SKIP fill:#2d3748,color:#e2e8f0
    style REINDEX fill:#744210,color:#fefcbf
    style PRUNE fill:#742a2a,color:#fed7d7
```

**Performance** (1300+ files, 1 file changed):

| Metric                              | Result                              |
| ----------------------------------- | ----------------------------------- |
| Unchanged files — hash computations | **0** (mtime/size fast-path)        |
| Changed file — embed + insert       | **~3 seconds**                      |
| No changes — full scan              | **instant** ("Already up to date!") |
| Deleted file — prune + scan         | **instant**                         |

### How pruning works

When a file is deleted or moved to an excluded directory (e.g. `_legacy/`), the next incremental `index_documents` call will:

1. Detect the file is missing from the current directory scan
2. Delete its vectors from Milvus (`filter: path == '...'`)
3. Remove it from the tracking file
4. Return a message: `"Pruned N deleted/moved files."`

No manual cleanup needed — just delete the file and re-index.

## Shell Reindex CLI

For large-scale indexing (1000+ files), use `reindex.py` directly for real-time logs and better error handling:

```bash
cd /path/to/mcp-markdown-rag

# Incremental (changed files only)
EMBEDDING_PROVIDER=vertex \
MILVUS_ADDRESS=http://localhost:19530 \
GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json \
VERTEX_PROJECT=your-project-id \
VERTEX_LOCATION=us-central1 \
uv run python reindex.py /path/to/vault

# Full rebuild (drop + re-create collection)
uv run python reindex.py /path/to/vault --force
```

Features over MCP `index_documents`:
- Real-time progress logs (batch N/M, elapsed time)
- 429 rate-limit retry with exponential backoff (5 attempts)
- Chunked Milvus insert (configurable via `MILVUS_INSERT_BATCH`)
- Non-recursive mode (`--no-recursive`)

## Configuration

### Core

| Variable             | Default                  | Description                                                          |
| -------------------- | ------------------------ | -------------------------------------------------------------------- |
| `EMBEDDING_PROVIDER` | `local`                  | `gemini`, `openai`, `openai-compatible`, `vertex`, `voyage`, `local` |
| `EMBEDDING_MODEL`    | (provider default)       | Model name override                                                  |
| `EMBEDDING_DIM`      | `768`                    | Vector dimension                                                     |
| `MILVUS_ADDRESS`     | `.db/milvus_markdown.db` | Milvus address (`http://host:port`) or local file path               |

### Indexing Tuning

| Variable                       | Default | Description                                                                |
| ------------------------------ | ------- | -------------------------------------------------------------------------- |
| `MARKDOWN_CHUNK_SIZE`          | `2048`  | Token chunk size for splitting documents                                   |
| `MARKDOWN_CHUNK_OVERLAP`       | `100`   | Token overlap between chunks                                               |
| `EMBEDDING_BATCH_SIZE`         | `250`   | Texts per embedding API call                                               |
| `EMBEDDING_BATCH_DELAY_MS`     | `0`     | Delay between embedding batches (ms). Set to `1000` for rate-limited APIs. |
| `EMBEDDING_CONCURRENT_BATCHES` | `4`     | Parallel embedding batches                                                 |
| `MILVUS_INSERT_BATCH`          | `5000`  | Rows per Milvus insert call (gRPC 64MB limit)                              |

### Exclusions

| Variable                 | Default | Description                                                                                                                                    |
| ------------------------ | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `MARKDOWN_EXCLUDE_DIRS`  | —       | Extra directories to exclude (comma-separated). Added to built-in: `node_modules`, `__pycache__`, `devlog`, `_legacy`, `dist`, `build`, `.git` |
| `MARKDOWN_EXCLUDE_FILES` | —       | Extra files to exclude (comma-separated). Added to built-in: `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`                                             |

### Provider Auth

| Variable                         | Description                                 |
| -------------------------------- | ------------------------------------------- |
| `GEMINI_API_KEY`                 | Gemini API key                              |
| `OPENAI_API_KEY`                 | OpenAI API key                              |
| `VOYAGE_API_KEY`                 | Voyage API key                              |
| `EMBEDDING_API_KEY`              | OpenAI-compatible API key                   |
| `EMBEDDING_BASE_URL`             | OpenAI-compatible base URL                  |
| `GOOGLE_APPLICATION_CREDENTIALS` | Service account JSON path for Vertex AI     |
| `VERTEX_PROJECT`                 | GCP project ID (auto-detected if SA has it) |
| `VERTEX_LOCATION`                | Vertex AI region (default: `us-central1`)   |

### Vertex AI Example

```json
{
  "mcpServers": {
    "markdown-rag": {
      "command": "uv",
      "args": ["--directory", "/path/to/mcp-markdown-rag", "run", "server.py"],
      "env": {
        "EMBEDDING_PROVIDER": "vertex",
        "EMBEDDING_MODEL": "gemini-embedding-001",
        "EMBEDDING_DIM": "768",
        "MARKDOWN_CHUNK_SIZE": "2048",
        "MARKDOWN_CHUNK_OVERLAP": "120",
        "EMBEDDING_BATCH_SIZE": "100",
        "EMBEDDING_BATCH_DELAY_MS": "1000",
        "EMBEDDING_CONCURRENT_BATCHES": "3",
        "MILVUS_INSERT_BATCH": "5000",
        "MILVUS_ADDRESS": "http://localhost:19530",
        "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/service-account.json",
        "VERTEX_PROJECT": "your-gcp-project-id",
        "VERTEX_LOCATION": "us-central1"
      }
    }
  }
}
```

## Debugging

```bash
npx @modelcontextprotocol/inspector uv --directory /path/to/mcp-markdown-rag run server.py
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

### About

This project is a fork of [MCP-Markdown-RAG](https://github.com/Zackriya-Solutions/MCP-Markdown-RAG) by Zackriya Solutions, heavily extended for production use.

**Key additions over upstream**:
- Multi-provider embeddings (Vertex AI, Gemini, OpenAI, Voyage)
- Single-pass incremental indexing with mtime/size fast-path
- Stale vector pruning for deleted/moved files
- Batch embedding with 429 retry + batch Milvus insert (gRPC 64MB limit)
- Shell reindex CLI (`reindex.py`) with real-time progress
- Configurable file/directory exclusions
- Milvus Standalone (Docker) support for multi-agent concurrent access
- Search results with relevance scores and file paths
