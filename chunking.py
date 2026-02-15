"""
chunking.py — Small chunk merge + parent header prefix injection.

Post-processes LlamaIndex nodes to:
1. Merge adjacent small chunks (< MIN_CHUNK_TOKENS) within the same file.
2. Inject parent header path as a prefix for richer search context.

Used by both server.py and reindex.py to prevent pipeline drift.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_index.core.schema import TextNode

# ---------------------------------------------------------------------------
# Token counting (tiktoken via llama-index transitive dep, fallback len//4)
# ---------------------------------------------------------------------------

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        """Count tokens using tiktoken cl100k_base (same as TokenTextSplitter)."""
        return len(_ENCODER.encode(text))

except Exception:
    def count_tokens(text: str) -> int:  # type: ignore[misc]
        """Fallback: rough estimate ~4 chars per token."""
        return len(text) // 4


# ---------------------------------------------------------------------------
# Header-only detection
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r"^#{1,6}\s")


def is_header_only(text: str) -> bool:
    """True if *every* non-blank line is a markdown header."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return True
    return all(_HEADER_RE.match(line) for line in lines)


# ---------------------------------------------------------------------------
# Merge small chunks
# ---------------------------------------------------------------------------


def _merge_text(a: "TextNode", b: "TextNode") -> None:
    """Merge b into a (text + metadata update)."""
    a.text = a.text.rstrip("\n") + "\n\n" + b.text.lstrip("\n")


def merge_small_chunks(
    nodes: list["TextNode"],
    min_tokens: int,
    max_tokens: int,
) -> list["TextNode"]:
    """Merge adjacent small chunks within same file.

    Rules:
    - Accumulator < min_tokens  → merge with next sibling
    - Merged result > max_tokens → emit accumulator, start new
    - Last node is header-only  → append to previous emitted
    - Single-chunk file         → always keep (no data loss)
    """
    if min_tokens <= 0:
        return nodes

    # Group by file_path
    file_groups: dict[str, list["TextNode"]] = {}
    for node in nodes:
        path = node.metadata.get("file_path", "")
        file_groups.setdefault(path, []).append(node)

    result: list["TextNode"] = []

    for _path, group in file_groups.items():
        if len(group) <= 1:
            # Single-chunk file — always keep
            result.extend(group)
            continue

        emitted: list["TextNode"] = []
        acc = group[0]

        for node in group[1:]:
            acc_tokens = count_tokens(acc.text)
            node_tokens = count_tokens(node.text)

            if acc_tokens < min_tokens:
                # Accumulator is small — merge unconditionally if within max
                if acc_tokens + node_tokens <= max_tokens:
                    _merge_text(acc, node)
                    continue
                # Would exceed max — emit acc, start fresh
                emitted.append(acc)
                acc = node
            elif acc_tokens + node_tokens <= max_tokens and is_header_only(node.text):
                # Next is header-only and fits — absorb it
                _merge_text(acc, node)
            else:
                # Accumulator is big enough — emit and move on
                emitted.append(acc)
                acc = node

        # Handle last accumulator
        if is_header_only(acc.text) and emitted:
            # Header-only tail → append to previous emitted chunk
            _merge_text(emitted[-1], acc)
        else:
            emitted.append(acc)

        result.extend(emitted)

    return result


# ---------------------------------------------------------------------------
# Inject parent header prefix
# ---------------------------------------------------------------------------


def inject_header_prefix(nodes: list["TextNode"]) -> list["TextNode"]:
    """Prepend parent header path to each chunk for richer search context.

    MarkdownNodeParser stores hierarchy in metadata['header_path'], e.g.:
      '/Title/Section A/Sub Section/'

    This converts it to markdown headers and prepends:
      '# Title\\n## Section A\\n### Sub Section\\n---\\n'

    Skips injection if the text already starts with the same header.
    """
    for node in nodes:
        header_path = node.metadata.get("header_path", "")
        if not header_path or header_path == "/":
            continue

        # Parse path segments: '/Title/Section A/' → ['Title', 'Section A']
        segments = [s for s in header_path.strip("/").split("/") if s]
        if not segments:
            continue

        # Build markdown header lines
        prefix_lines = []
        for i, seg in enumerate(segments):
            level = min(i + 1, 6)  # h1, h2, h3, ..., h6 max
            prefix_lines.append(f"{'#' * level} {seg}")

        prefix = "\n".join(prefix_lines) + "\n---\n"

        # Skip if text already starts with the same first header
        first_header = prefix_lines[0]
        if node.text.lstrip().startswith(first_header):
            continue

        node.text = prefix + node.text

    return nodes
