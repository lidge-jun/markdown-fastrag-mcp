#!/usr/bin/env python3
"""
markdown-rag 리인덱싱 스크립트 (shell 직접 실행용)
MCP 서버 없이 직접 Milvus에 인덱싱, 로그 출력

환경 변수는 MCP config와 동일하게 설정 필요:
  EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_DIM,
  MILVUS_ADDRESS, GOOGLE_APPLICATION_CREDENTIALS 등
"""
import asyncio
import os
import sys
import time

# 프로젝트 루트를 path에 추가
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

# env 기본값 (server.py와 동일 — 없으면 server.py 기본값 사용)
os.environ.setdefault("EMBEDDING_BATCH_SIZE", "100")
os.environ.setdefault("EMBEDDING_CONCURRENT_BATCHES", "3")
os.environ.setdefault("EMBEDDING_BATCH_DELAY_MS", "1000")

# === 임포트 (server.py가 env 기반으로 provider 자동 선택) ===
from server import (
    embedding_fn, milvus_client, COLLECTION_NAME,
    MARKDOWN_CHUNK_SIZE, MARKDOWN_CHUNK_OVERLAP,
    EMBEDDING_BATCH_SIZE, EMBEDDING_CONCURRENT_BATCHES,
    EMBEDDING_BATCH_DELAY_MS, EMBEDDING_PROVIDER,
    MIN_CHUNK_TOKENS,
)
from utils import ensure_collection, get_index_delta, list_md_files, update_tracking_file
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.text_splitter import TokenTextSplitter
from concurrent.futures import ThreadPoolExecutor

# Milvus gRPC 64MB 제한 대비 insert 배치 크기
MILVUS_INSERT_BATCH = int(os.getenv("MILVUS_INSERT_BATCH", "5000"))


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


async def reindex(target_path: str, recursive: bool = True, force: bool = False):
    log(f"🚀 Reindex 시작: {target_path}")
    log(f"   recursive={recursive}, force={force}")
    log(f"   Milvus: {os.getenv('MILVUS_ADDRESS', 'local')}")
    log(f"   Embedding: {EMBEDDING_PROVIDER}")
    log(f"   chunk_size={MARKDOWN_CHUNK_SIZE}, overlap={MARKDOWN_CHUNK_OVERLAP}")
    log(f"   batch_size={EMBEDDING_BATCH_SIZE}, concurrent={EMBEDDING_CONCURRENT_BATCHES}")
    log(f"   insert_batch={MILVUS_INSERT_BATCH}")
    print()

    # 1. 파일 수집
    if force:
        log("🗑️  Force mode: 기존 컬렉션 드롭 후 재생성...")
        if milvus_client.has_collection(COLLECTION_NAME):
            milvus_client.drop_collection(COLLECTION_NAME)
        ensure_collection(milvus_client)

        all_files = list_md_files(target_path, recursive=recursive)
        log(f"📄 마크다운 파일 {len(all_files)}개 발견")
        if not all_files:
            log("❌ 인덱싱할 파일 없음!")
            return

        documents = SimpleDirectoryReader(
            input_files=all_files, required_exts=[".md"]
        ).load_data()
        processed_files = [doc.metadata["file_path"] for doc in documents]
    else:
        ensure_collection(milvus_client)

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
        log(f"📄 변경된 파일 {len(changed_files)}개:")
        for f in changed_files[:20]:
            log(f"   {os.path.basename(f)}")
        if len(changed_files) > 20:
            log(f"   ... 외 {len(changed_files) - 20}개")

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

    # 2. Pre-process: strip YAML frontmatter before chunking
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

    # 3. 청킹
    log(f"✂️  청킹 중... (documents={len(documents)})")
    nodes = MarkdownNodeParser(chunk_size=MARKDOWN_CHUNK_SIZE).get_nodes_from_documents(documents)
    chunk_overlap = min(MARKDOWN_CHUNK_OVERLAP, max(0, MARKDOWN_CHUNK_SIZE - 1))
    chunked_nodes = TokenTextSplitter(
        chunk_size=MARKDOWN_CHUNK_SIZE, chunk_overlap=chunk_overlap
    ).get_nodes_from_documents(nodes)
    chunked_nodes = [node for node in chunked_nodes if node.text.strip()]

    # Post-process: merge small chunks + inject parent header context.
    if MIN_CHUNK_TOKENS > 0:
        pre_merge = len(chunked_nodes)
        chunked_nodes = merge_small_chunks(
            chunked_nodes, MIN_CHUNK_TOKENS, MARKDOWN_CHUNK_SIZE
        )
        if pre_merge != len(chunked_nodes):
            log(f"   → 병합: {pre_merge} → {len(chunked_nodes)} 청크")
    chunked_nodes = inject_header_prefix(chunked_nodes)
    log(f"   → {len(chunked_nodes)} 청크 생성 (min_tokens={MIN_CHUNK_TOKENS})")

    # 3. 임베딩
    texts = [node.text for node in chunked_nodes]
    batches = [texts[i:i + EMBEDDING_BATCH_SIZE] for i in range(0, len(texts), EMBEDDING_BATCH_SIZE)]
    total_batches = len(batches)
    log(f"🧠 임베딩 시작: {len(texts)} 청크, {total_batches} 배치")

    vectors = [None] * len(texts)
    executor = ThreadPoolExecutor(max_workers=EMBEDDING_CONCURRENT_BATCHES)
    loop = asyncio.get_event_loop()
    t0 = time.time()

    async def embed_one(batch_idx, batch, offset):
        log(f"   배치 {batch_idx+1}/{total_batches} ({offset}~{offset+len(batch)}/{len(texts)})")
        for retry in range(5):
            try:
                result = await loop.run_in_executor(executor, embedding_fn.encode_documents, batch)
                for j, vec in enumerate(result):
                    vectors[offset + j] = vec
                return
            except Exception as e:
                if "429" in str(e) and retry < 4:
                    wait = (2 ** retry) * 5  # 5s, 10s, 20s, 40s
                    log(f"   ⚠️  429 Rate Limit! {wait}s 대기 후 재시도 ({retry+1}/5)")
                    await asyncio.sleep(wait)
                else:
                    raise

    for wave_start in range(0, total_batches, EMBEDDING_CONCURRENT_BATCHES):
        wave = []
        for k in range(wave_start, min(wave_start + EMBEDDING_CONCURRENT_BATCHES, total_batches)):
            offset = k * EMBEDDING_BATCH_SIZE
            wave.append(embed_one(k, batches[k], offset))
        await asyncio.gather(*wave)
        if EMBEDDING_BATCH_DELAY_MS > 0:
            await asyncio.sleep(EMBEDDING_BATCH_DELAY_MS / 1000.0)

    executor.shutdown(wait=False)
    elapsed = time.time() - t0
    log(f"   → 임베딩 완료 ({elapsed:.1f}s)")

    # 4. Milvus 삽입 (gRPC 64MB 제한 대비 배치 처리)
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

    total_inserted = 0
    insert_batches = [data[i:i + MILVUS_INSERT_BATCH] for i in range(0, len(data), MILVUS_INSERT_BATCH)]
    log(f"💾 Milvus 삽입 중... ({len(data)} 청크, {len(insert_batches)} 배치)")

    for idx, batch in enumerate(insert_batches):
        res = milvus_client.insert(collection_name=COLLECTION_NAME, data=batch)
        count = res.get("insert_count", len(batch))
        total_inserted += count
        log(f"   배치 {idx+1}/{len(insert_batches)}: {count}건 ({total_inserted}/{len(data)})")

    # 5. 트래킹 업데이트
    update_tracking_file(processed_files)

    total_time = time.time() - t0
    print()
    log(f"🎉 완료!")
    log(f"   파일: {len(processed_files)}개")
    log(f"   청크: {len(chunked_nodes)}개")
    log(f"   삽입: {total_inserted}건")
    log(f"   총 소요: {total_time:.1f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="markdown-rag 리인덱싱 (shell 직접 실행)")
    parser.add_argument("path", nargs="?", default=os.getcwd(),
                        help="인덱싱할 디렉토리 (기본: 현재 디렉토리)")
    parser.add_argument("-f", "--force", action="store_true", help="전체 재인덱싱 (기존 데이터 삭제)")
    parser.add_argument("--no-recursive", action="store_true", help="하위 디렉토리 제외")
    args = parser.parse_args()

    asyncio.run(reindex(args.path, recursive=not args.no_recursive, force=args.force))
