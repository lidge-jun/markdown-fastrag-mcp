# Modified from original MCP-Markdown-RAG by Zackriya Solutions
# (https://github.com/Zackriya-Solutions/MCP-Markdown-RAG)
# Changes: configurable embedding dimension via EMBEDDING_DIM env var.

import hashlib
import json
import os

from pydantic import BaseModel
from pymilvus import MilvusClient

INDEX_DATA_PATH = os.getenv("MARKDOWN_RAG_CACHE_DIR", "./.db")
INDEX_TRACKING_FILE = "index_tracking.json"
COLLECTION_NAME = os.getenv("MARKDOWN_COLLECTION", "markdown_vectors")


class Entity(BaseModel):
    text: str
    filename: str
    path: str
    tags: str = ""
    aliases: str = ""


class SearchResult(BaseModel):
    id: int
    distance: float
    entity: Entity


def load_tracking_file():
    try:
        with open(os.path.join(INDEX_DATA_PATH, INDEX_TRACKING_FILE), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_tracking_file(tracking_data):
    with open(os.path.join(INDEX_DATA_PATH, INDEX_TRACKING_FILE), "w") as f:
        json.dump(tracking_data, f, indent=2)


def get_file_info(file_path):
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    file_stat = os.stat(file_path)
    return file_hash, file_stat.st_mtime, file_stat.st_size


def _get_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _parse_tracking_entry(entry):
    """
    Backward-compatible parser for tracking entries.
    Supports legacy [hash, mtime] and new [hash, mtime, size] formats.
    """
    if isinstance(entry, (list, tuple)):
        file_hash = entry[0] if len(entry) > 0 else None
        modified_time = entry[1] if len(entry) > 1 else None
        file_size = entry[2] if len(entry) > 2 else None
        return file_hash, modified_time, file_size

    if isinstance(entry, dict):
        return entry.get("hash"), entry.get("mtime"), entry.get("size")

    return entry, None, None


def update_tracking_file(processed_files: list[str], is_clear: bool = False):
    tracking_data = load_tracking_file()
    if is_clear:
        save_tracking_file({})
        return
    for file_path in processed_files:
        try:
            tracking_data[file_path] = get_file_info(file_path)
        # Remove deleted files from tracking
        except (FileNotFoundError, PermissionError):
            tracking_data.pop(file_path, None)
    save_tracking_file(tracking_data)


_DEFAULT_EXCLUDE_DIRS = {"node_modules", "__pycache__", "devlog", "_legacy", "dist", "build", ".git"}
_extra = os.getenv("MARKDOWN_EXCLUDE_DIRS", "")
EXCLUDE_DIRS = _DEFAULT_EXCLUDE_DIRS | (set(_extra.split(",")) if _extra else set())

_DEFAULT_EXCLUDE_FILES = {"AGENTS.md", "CLAUDE.md", "GEMINI.md"}
_extra_files = os.getenv("MARKDOWN_EXCLUDE_FILES", "")
EXCLUDE_FILES = _DEFAULT_EXCLUDE_FILES | (set(_extra_files.split(",")) if _extra_files else set())


def list_md_files(base_dir: str, recursive: bool = False) -> list[str]:
    md_files = []
    if recursive:
        for root, dirs, files in os.walk(base_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in EXCLUDE_DIRS]
            for file in files:
                if file.endswith(".md") and file not in EXCLUDE_FILES:
                    md_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(base_dir):
            if file.endswith(".md") and file not in EXCLUDE_FILES:
                md_files.append(os.path.join(base_dir, file))
    return md_files


def get_changed_files(
    directory: str,
    recursive: bool = False,
) -> list[str]:
    """
    Get list of changed file paths in the directory.
    """
    changed_files, _ = get_index_delta(directory, recursive)
    return changed_files


def get_deleted_files(
    directory: str,
    recursive: bool = False,
) -> list[str]:
    """
    Detect files that were tracked but no longer exist on disk
    (deleted or moved to excluded directories like _legacy/).
    Returns list of stale file paths and removes them from tracking.
    """
    _, deleted_files = get_index_delta(directory, recursive)
    return deleted_files


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
    # IMPORTANT: Only prune files that fall UNDER the target directory.
    # Files outside target_path belong to other index scopes and must not be touched.
    target_prefix = os.path.normpath(directory) + os.sep
    for tracked_path in list(tracking_data.keys()):
        if tracked_path.startswith(target_prefix) and tracked_path not in current_files_set:
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


EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))


def ensure_collection(milvus_client: MilvusClient):
    if milvus_client.has_collection(COLLECTION_NAME):
        return
    milvus_client.create_collection(COLLECTION_NAME, dimension=EMBEDDING_DIM, auto_id=True)
