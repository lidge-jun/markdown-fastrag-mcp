# reindex.py: Shell 직접 실행 리인덱싱 스크립트

**Date**: 2026-02-15
**Status**: Complete

## 배경

MCP `index_documents()` 도구의 한계:
- AI가 실시간 로그를 볼 수 없음 (MCP는 최종 return만 전달)
- 타임아웃 위험 (23791 청크 임베딩 ~7분 소요)
- 에러 발생 시 디버깅 어려움

→ shell에서 직접 실행하는 `reindex.py` 스크립트 작성

## 발생한 이슈 & 해결

### Issue 1: Vertex AI 429 Rate Limit

- **증상**: 배치 72/119에서 `HTTP Error 429: Too Many Requests`
- **원인**: `batch_size=200, delay=0ms` → Vertex 분당 토큰 쿼터 초과
- **해결**: `batch_size=100, delay=1000ms` + 429 재시도 로직 (exponential backoff, 5회)

```
EMBEDDING_BATCH_SIZE:     200 → 100
EMBEDDING_BATCH_DELAY_MS: 0   → 1000
```

4개 에이전트 MCP 설정 전부 동기화:
- `~/.gemini/antigravity/mcp_config.json`
- `~/.codex/config.toml`
- `~/Library/Application Support/Code/User/mcp.json`
- `~/.claude/settings.local.json`

### Issue 2: Milvus gRPC 64MB Message Limit

- **증상**: 임베딩 완료 후 insert에서 `RESOURCE_EXHAUSTED (92MB > 64MB)`
- **원인**: 23791 청크를 단일 gRPC call로 전송
- **해결**: `MILVUS_INSERT_BATCH=5000` (5배치, 각 ~18MB)

## reindex.py 특징

| 기능          | 설명                                        |
| ------------- | ------------------------------------------- |
| 429 재시도    | 5회 exponential backoff (5s, 10s, 20s, 40s) |
| insert 배치   | 5000건씩 분할 (gRPC 64MB 제한 회피)         |
| env 기반 설정 | Vertex 하드코딩 없음, provider 자동 선택    |
| 실시간 로그   | 배치 진행률, 에러, 소요 시간 출력           |
| 증분/전체     | `--force` 전체, 기본은 변경 파일만          |

## 사용법

```bash
cd 700_projects/markdown-fastrag-mcp

# 전체 재인덱싱
EMBEDDING_PROVIDER=vertex \
MILVUS_ADDRESS=http://localhost:19530 \
GOOGLE_APPLICATION_CREDENTIALS=~/secure/vertex-sa.json \
VERTEX_PROJECT=gen-lang-client-0239871193 \
VERTEX_LOCATION=us-central1 \
uv run python reindex.py /path/to/vault --force

# 증분 (변경 파일만)
uv run python reindex.py /path/to/vault
```

## 성능 기록 (1481 파일, 23791 청크)

| 단계                           | 시간    |
| ------------------------------ | ------- |
| 파일 수집                      | ~1s     |
| 청킹                           | ~6s     |
| 임베딩 (238배치, concurrent=3) | ~390s   |
| Milvus insert (5배치)          | TBD     |
| **총 예상**                    | ~7-8min |

## 변경 파일

- `reindex.py` — 신규 작성
- `server.py` — EMBEDDING_PROVIDER export 추가

## TODO

- [x] `server.py` index_documents에도 insert 배치 처리 적용 → Phase 2에서 완료
- [ ] smart-coding-mcp에도 유사한 `reindex.js` CLI 작성
- [x] rag skill 문서 업데이트 → `.agents/skills/rag/SKILL.md` 에 shell reindex 섹션 추가 완료
