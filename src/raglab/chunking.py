"""Heading-aware chunking.

Design notes
------------
* Documents are split on markdown headings first, because a section is the
  smallest unit that is still self-contained for a reader.
* Long sections are then packed into overlapping windows at sentence
  boundaries, so no chunk ever ends mid-sentence.
* Every chunk keeps its heading path ("Billing > Refunds > Timing"), which is
  prepended when indexing. This is the cheapest known win for retrieval
  quality on structured business documents.
"""

from __future__ import annotations

import re

from .text import TABLE_ROW_SEPARATOR, split_sentences
from .types import Chunk

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def expand_markdown_tables(text: str) -> str:
    """Rewrite markdown table rows as ``Header: value`` lines.

    A bare row like ``| Growth | $199 | 10 |`` is close to meaningless: neither a
    reader nor a retriever can tell which number is the price. Folding the header
    into every row makes each row self-contained, which measurably improves both
    retrieval and answer quality on pricing/limit tables. Separator rows are
    dropped and the header line is rewritten as a plain sentence so that it does
    not masquerade as an answer.
    """
    output: list[str] = []
    header: list[str] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        is_row = len(stripped) > 1 and stripped.startswith("|") and stripped.endswith("|")
        if not is_row:
            header = None
            output.append(line)
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(cell and set(cell) <= set("-: ") for cell in cells):
            continue  # | --- | --- | separator
        if header is None:
            header = cells
            output.append(f"Table columns: {', '.join(cells)}.")
            continue
        pairs = [
            f"{header[i] if i < len(header) else f'column {i + 1}'}: {cell}"
            for i, cell in enumerate(cells)
            if cell
        ]
        output.append(TABLE_ROW_SEPARATOR.join(pairs))
    return "\n".join(output)


def _windows(sentences: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    """Greedily pack sentences into windows of at most `max_chars`."""
    windows: list[list[str]] = []
    current: list[str] = []
    size = 0
    for sentence in sentences:
        addition = len(sentence) + 1
        if current and size + addition > max_chars:
            windows.append(current)
            # carry the tail of the previous window over for context continuity
            carried: list[str] = []
            carried_size = 0
            for prev in reversed(current):
                if carried_size + len(prev) > overlap_chars:
                    break
                carried.insert(0, prev)
                carried_size += len(prev) + 1
            current = list(carried)
            size = carried_size
        current.append(sentence)
        size += addition
    if current:
        windows.append(current)
    # Join with newlines: sentence-per-line keeps markdown tables readable
    # (a table row stays a row instead of being flattened into prose).
    return ["\n".join(w) for w in windows if w]


def chunk_markdown(
    text: str,
    doc_id: str,
    doc_title: str = "",
    max_chars: int = 700,
    overlap_chars: int = 120,
) -> list[Chunk]:
    """Split a markdown document into heading-scoped, sentence-aligned chunks."""
    sections: list[tuple[tuple[str, ...], str]] = []
    path: list[str] = []
    current_path: tuple[str, ...] = ()
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((current_path, body))

    for line in expand_markdown_tables(text).splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            buffer = []
            level = len(match.group(1))
            title = match.group(2).strip()
            path = path[: level - 1] + [title]
            current_path = tuple(path)
        else:
            buffer.append(line)
    flush()

    chunks: list[Chunk] = []
    for heading_path, body in sections:
        heading = " > ".join(heading_path)
        # A section's own first line often repeats the heading; keep it, it is harmless.
        sentences = split_sentences(body)
        for window in _windows(sentences, max_chars, overlap_chars):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{len(chunks):03d}",
                    doc_id=doc_id,
                    doc_title=doc_title or doc_id,
                    heading=heading,
                    text=window.strip(),
                    index_in_doc=len(chunks),
                )
            )
    return chunks
