# Phase 4: Stale Vector Pruning + Incremental Indexing Optimization

> **상태**: ✅ 완료 (구현 + 검증)
> **날짜**: 2026-02-15
> **우선순위**: 높음 (데이터 정합성 + 성능)

---

## 1. 문제 (2가지)

### 문제 A: 삭제된 파일의 유령 벡터

파일을 삭제하거나 `_legacy/` 등 제외 디렉토리로 이동하면, **Milvus 벡터가 그대로 남아서 검색 결과에 유령 문서가 계속 나온다.**

```
1. test.md 생성 → index_documents → 검색에 나옴 ✅
2. test.md 삭제
3. index_documents 다시 실행
4. 검색하면 삭제된 test.md가 여전히 나옴 ← 버그 ❌
```

**원인**: `get_changed_files()`가 디스크에 있는 파일만 순회. 사라진 파일은 비교 대상에서 빠지고, Milvus에서 벡터를 삭제하는 코드가 없었음.

### 문제 B: 증분 인덱싱이 느림

파일 1개만 변경해도 인덱싱이 수 초~수십 초 걸림.

**원인**: `get_changed_files()`가 매번 **모든 파일의 MD5 해시를 계산** (파일 전체 읽기). 1326개 파일 × 파일 읽기 = 불필요한 I/O.

smart-coding-mcp는 mtime/size 메타데이터만으로 빠르게 비교하고, 해시는 메타데이터가 바뀐 파일에만 계산.

---

## 2. 구현

### 2-1. `utils.py` — 단일 스캔 델타 함수

**핵심 변경**: `get_changed_files()` + `get_deleted_files()` 2-pass 구조를 **`get_index_delta()` 1-pass**로 통합.

```python
def get_index_delta(
    directory: str,
    recursive: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Single-pass incremental diff:
    - changed_files: files that need re-indexing
    - deleted_files: tracked files no longer present (or newly excluded)
    """
    tracking_data = load_tracking_file()
    changed_files: list[str] = []
    deleted_files: list[str] = []
    tracking_dirty = False

    md_files = list_md_files(directory, recursive)
    current_files_set = set(md_files)

    # Step 1: prune tracked files missing from current scan.
    for tracked_path in list(tracking_data.keys()):
        if tracked_path not in current_files_set:
            deleted_files.append(tracked_path)
            tracking_data.pop(tracked_path, None)
            tracking_dirty = True

    # Step 2: detect changed/new files with fast metadata check first.
    for file_path in md_files:
        if file_path not in tracking_data:
            changed_files.append(file_path)
            continue

        stored_hash, stored_time, stored_size = _parse_tracking_entry(
            tracking_data[file_path]
        )
        if not stored_hash:
            changed_files.append(file_path)
            continue

        try:
            file_stat = os.stat(file_path)
        except (FileNotFoundError, PermissionError):
            tracking_data.pop(file_path, None)
            deleted_files.append(file_path)
            tracking_dirty = True
            continue

        current_modified_time = file_stat.st_mtime
        current_size = file_stat.st_size

        # Fast path: unchanged metadata => unchanged content (skip file read/hash).
        if stored_time == current_modified_time and (
            stored_size is None or stored_size == current_size
        ):
            # Migrate legacy [hash, mtime] entries without forcing reindex.
            if stored_size is None:
                tracking_data[file_path] = [stored_hash, stored_time, current_size]
                tracking_dirty = True
            continue

        current_hash = _get_file_hash(file_path)
        if current_hash != stored_hash:
            changed_files.append(file_path)
            continue

        # Metadata changed but content same: refresh tracking only.
        tracking_data[file_path] = [current_hash, current_modified_time, current_size]
        tracking_dirty = True

    if tracking_dirty:
        save_tracking_file(tracking_data)

    return changed_files, deleted_files
```

**최적화 포인트 3가지**:

| 최적화        | Before                     | After                      | 효과                       |
| ------------- | -------------------------- | -------------------------- | -------------------------- |
| 스캔 횟수     | 2-pass (changed + deleted) | 1-pass (`get_index_delta`) | 디렉토리 순회 1회로 감소   |
| 해시 계산     | 모든 파일마다 MD5          | mtime/size 동일하면 스킵   | 무변경 시 해시 계산 0회    |
| tracking 포맷 | `[hash, mtime]`            | `[hash, mtime, size]`      | size 비교로 추가 fast-path |

### 2-2. `utils.py` — tracking 포맷 하위호환

기존 `[hash, mtime]`과 신규 `[hash, mtime, size]` 모두 파싱 가능:

```python
def _parse_tracking_entry(entry):
    """Backward-compatible parser for tracking entries."""
    if isinstance(entry, (list, tuple)):
        file_hash = entry[0] if len(entry) > 0 else None
        modified_time = entry[1] if len(entry) > 1 else None
        file_size = entry[2] if len(entry) > 2 else None
        return file_hash, modified_time, file_size
    if isinstance(entry, dict):
        return entry.get("hash"), entry.get("mtime"), entry.get("size")
    return entry, None, None
```

기존 tracking 데이터는 다음 스캔 때 자동으로 size 정보가 보강됨 (reindex 불필요).

### 2-3. `utils.py` — `get_file_info()` 확장

저장 시 size도 포함:

```python
def get_file_info(file_path):
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    file_stat = os.stat(file_path)
    return file_hash, file_stat.st_mtime, file_stat.st_size
```

### 2-4. `server.py` — 삭제 벡터 정리

```python
# Single-pass delta scan (changed + deleted) for faster incremental indexing.
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

if not changed_files:
    if pruned_count > 0:
        return {"message": f"Pruned {pruned_count} deleted/moved files. No new files to index."}
    return {"message": "Already up to date, Nothing to index!"}
```

### 2-5. `reindex.py` — 동일 패턴 적용

```python
# Single-pass delta scan (changed + deleted) for faster incremental indexing.
changed_files, deleted_files = get_index_delta(target_path, recursive=recursive)
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
    log(f"🗑️  Pruned {pruned_count} deleted/moved files from index")

if not changed_files:
    if pruned_count > 0:
        log(f"✅ Pruned {pruned_count} stale files. No new changes.")
    else:
        log("✅ Already up to date!")
    return
```

---

## 3. 하위호환성

| 항목                     | 영향                                                       |
| ------------------------ | ---------------------------------------------------------- |
| `get_changed_files()`    | 내부적으로 `get_index_delta()` 위임. 기존 호출부 호환 유지 |
| `get_deleted_files()`    | 동일하게 `get_index_delta()` 위임                          |
| tracking 포맷            | `[hash, mtime]` → `[hash, mtime, size]` 자동 마이그레이션  |
| `index_documents()` 응답 | 기존 메시지 형식 유지, pruned 정보만 추가                  |
| `force_reindex=True`     | 영향 없음 (컬렉션 전체 drop 후 재생성)                     |

---

## 4. 추가 발견: MCP 경로 버그

### 증상

MCP `index_documents()` 호출 시 1개 파일 변경인데도 **수십 초간 행(hang)**.
Shell `reindex.py`는 동일 작업 3초 완료.

### 원인

`~/.gemini/antigravity/mcp_config.json`의 `--directory`가 구 iCloud 경로를 가리키고 있었음:

```diff
- "/Users/jun/Library/Mobile Documents/iCloud~md~obsidian/Documents/new/700_projects/markdown-fastrag-mcp"
+ "/Users/jun/Developer/new/700_projects/markdown-fastrag-mcp"
```

구 iCloud 경로의 `.db/index_tracking.json`이 **빈 파일** (0줄) → MCP 서버가 매번 1326개 파일 전체를 신규로 인식 → 전체 임베딩 시도 → 타임아웃.

### 수정

`mcp_config.json`의 경로를 `/Users/jun/Developer/new/...`로 수정. 수정 후 MCP도 shell과 동일 속도.

---

## 5. 테스트 결과

### Shell 테스트 (reindex.py)

```
Step 1: help.md 생성 → 인덱싱    → ✅ 1파일 3청크, 2.5s
Step 2: 시맨틱 검색               → ✅ 91.5% relevance
Step 3: help.md 삭제              → ✅
Step 4: 증분 인덱싱               → ✅ "Pruned 1 deleted/moved files"
Step 5: 검색 재확인               → ✅ help.md vectors: 0 (미검출)
```

### MCP 테스트 (index_documents / search_documents)

```
Step 1: mcp_test_canary.md 생성         → ✅
Step 2: MCP index_documents(force=false) → ✅ 2파일 17청크, incremental
Step 3: MCP search_documents             → ✅ 90.2% relevance
Step 4: 파일 삭제                         → ✅
Step 5: MCP index_documents(force=false) → ✅ "Pruned 1 deleted/moved files"
Step 6: MCP search_documents             → ✅ 미검출 (prune 성공)
```

### 성능 비교 (1326 파일, 삭제 1건)

| 측정                       | 결과                               |
| -------------------------- | ---------------------------------- |
| 무변경 재스캔 시 해시 계산 | **0회** (mtime/size fast-path)     |
| 1-pass vs 2-pass 스캔 시간 | 0.0079s vs 0.018s (**2.28x 개선**) |
| 1파일 변경 + 임베딩        | **2.5~3.3s**                       |
| 무변경 재실행              | **즉시** ("Already up to date!")   |

---

## 6. 변경 파일 요약

| 파일              | 변경 내용                                                                                                      |
| ----------------- | -------------------------------------------------------------------------------------------------------------- |
| `utils.py`        | `get_index_delta()` 신규, `_parse_tracking_entry()` 신규, `_get_file_hash()` 신규, `get_file_info()` size 추가 |
| `server.py`       | `index_documents()` → `get_index_delta()` + prune 루프                                                         |
| `reindex.py`      | 증분 경로 → `get_index_delta()` + prune 루프                                                                   |
| `mcp_config.json` | `--directory` 경로 수정 (iCloud → Developer/new)                                                               |

---

## 7. 남은 이슈

- [ ] `PRUNE_TEST_CANARY.md` orphan 벡터 1개 — tracking에서 이미 제거되어 prune 대상으로 감지 안 됨. 다음 `--force` 전체 리인덱싱 시 자동 해결.
- [ ] Milvus orphan 정리 유틸리티 — tracking에 없지만 Milvus에 남은 벡터를 찾아 정리하는 유틸. 우선순위 낮음.
