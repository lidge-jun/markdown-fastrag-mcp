# Phase 4: Stale Vector Pruning (삭제/이동 파일 벡터 정리)

> **상태**: 계획 수립
> **날짜**: 2026-02-15
> **우선순위**: 높음 (데이터 정합성 버그)

---

## 1. 문제

파일을 삭제하거나 `_legacy/` 등 제외 디렉토리로 이동하면, **Milvus 벡터가 그대로 남아서 검색 결과에 유령 문서가 계속 나온다.**

### 재현 순서
1. `test.md` 생성 → `index_documents(recursive=True)` → 검색에 나옴
2. `test.md` 삭제 (또는 `_legacy/`로 이동)
3. `index_documents(recursive=True)` 다시 실행
4. **검색하면 삭제된 `test.md` 내용이 여전히 나옴** ← 버그

### 원인 분석

`get_changed_files()` (utils.py:87)의 로직:

```python
md_files = list_md_files(directory, recursive)  # 현재 디스크에 있는 파일만 순회
for file_path in md_files:
    # 새 파일이거나 변경된 파일만 changed_files에 추가
```

- 디스크에서 사라진 파일은 `md_files`에 안 잡힘
- `tracking_data`에는 남아있지만 비교 대상에서 빠짐
- Milvus에서 해당 path의 벡터를 삭제하는 코드가 **없음**

### smart-coding-mcp와의 비교

smart-coding-mcp는 이 문제를 `index-codebase.js:656-677`에서 해결:

```js
// Step 1.5: Prune deleted or excluded files from cache
const currentFilesSet = new Set(files);
const cachedFiles = Array.from(this.cache.getAllFileHashes().keys());
for (const cachedFile of cachedFiles) {
  if (!currentFilesSet.has(cachedFile)) {
    this.cache.removeFileFromStore(cachedFile);
    this.cache.deleteFileHash(cachedFile);
    prunedCount++;
  }
}
```

---

## 2. 수정 계획

### 2-1. 수정 파일: `utils.py`

**`get_changed_files()` 함수에 삭제 파일 감지 로직 추가**

현재:
```python
def get_changed_files(directory, recursive=False):
    tracking_data = load_tracking_file()
    changed_files = []
    md_files = list_md_files(directory, recursive)
    for file_path in md_files:
        # ... 변경 감지만 함
    return changed_files
```

변경:
```python
def get_changed_files(directory, recursive=False):
    tracking_data = load_tracking_file()
    changed_files = []
    md_files = list_md_files(directory, recursive)
    current_files_set = set(md_files)

    for file_path in md_files:
        # ... 기존 변경 감지 로직 유지

    # 삭제/이동된 파일 감지
    deleted_files = []
    for tracked_path in list(tracking_data.keys()):
        if tracked_path not in current_files_set:
            deleted_files.append(tracked_path)
            tracking_data.pop(tracked_path, None)

    # tracking 파일 업데이트 (삭제된 항목 제거)
    if deleted_files:
        save_tracking_file(tracking_data)

    return changed_files, deleted_files  # 반환값 변경
```

### 2-2. 수정 파일: `server.py`

**`index_documents()` 함수에서 삭제 파일 벡터 정리**

현재:
```python
changed_files = get_changed_files(target_path, recursive=recursive)
if not changed_files:
    return {"message": "Already up to date, Nothing to index!"}
```

변경:
```python
changed_files, deleted_files = get_changed_files(target_path, recursive=recursive)

# 삭제/이동된 파일의 벡터 정리
pruned_count = 0
for file_path in deleted_files:
    try:
        milvus_client.delete(
            collection_name=COLLECTION_NAME, filter=f"path == '{file_path}'"
        )
        pruned_count += 1
    except Exception:
        continue

if not changed_files and not deleted_files:
    return {"message": "Already up to date, Nothing to index!"}

if not changed_files and deleted_files:
    return {"message": f"Pruned {pruned_count} deleted/moved files from index. No new files to index."}
```

### 2-3. 수정 파일: `reindex.py`

`reindex.py`에서도 `get_changed_files()`의 새 반환값을 처리해야 하는지 확인 필요.

---

## 3. 하위호환성

| 항목 | 영향 |
|------|------|
| `get_changed_files()` 반환값 | `list` → `tuple(list, list)` 변경. 호출부 전부 수정 필요 |
| `index_documents()` 응답 | 기존 메시지 형식 유지, pruned 정보만 추가 |
| `force_reindex=True` | 영향 없음 (컬렉션 전체 drop 후 재생성) |
| `clear_index` | 영향 없음 (전체 삭제) |

**대안**: 반환값 변경 대신 별도 함수 `get_deleted_files()` 추가도 가능. 호출부 수정 최소화.

---

## 4. 테스트 계획

```bash
# 1. 테스트 파일 생성
echo "# PRUNE_TEST_CANARY_9999" > /tmp/prune_test.md
cp /tmp/prune_test.md <워크스페이스>/prune_test.md

# 2. 인덱싱
index_documents(recursive=True)

# 3. 검색 확인 (나와야 함)
search_documents(query="PRUNE_TEST_CANARY_9999")

# 4. 파일 삭제
rm <워크스페이스>/prune_test.md

# 5. 재인덱싱
index_documents(recursive=True)
# 기대 응답: "Pruned 1 deleted/moved files from index..."

# 6. 검색 확인 (안 나와야 함)
search_documents(query="PRUNE_TEST_CANARY_9999")
```

---

## 5. 구현 순서

1. `utils.py` - `get_changed_files()` 수정 (삭제 파일 감지 + 반환)
2. `server.py` - `index_documents()` 수정 (삭제 벡터 정리)
3. `reindex.py` - 호출부 수정 (있다면)
4. 테스트 실행
5. `devlog/phase4-stale-vector-pruning.md` 결과 기록

**예상 수정량**: ~20줄 추가, ~5줄 변경
