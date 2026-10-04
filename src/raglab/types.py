"""Core data structures for the grounded RAG pipeline.

Everything here is plain dataclasses so that traces are easy to serialise,
print, and assert on in tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text, always tied back to its source document."""

    chunk_id: str
    doc_id: str
    doc_title: str
    heading: str  # "Billing > Refunds" - the heading path inside the document
    text: str
    index_in_doc: int

    @property
    def indexed_text(self) -> str:
        """Text used for indexing: the heading path gives short chunks context."""
        return f"{self.heading}\n{self.text}" if self.heading else self.text


@dataclass
class Retrieved:
    """One retrieval hit with the signals that produced it."""

    chunk: Chunk
    rank: int
    score: float  # fused ranking score (higher is better)
    similarity: float = 0.0  # absolute vector similarity (interpretable)
    query_coverage: float = 0.0  # share of the question's terms present in the chunk
    sources: tuple[str, ...] = ()  # which retrievers surfaced it: ("bm25", "vector")

    @property
    def label(self) -> str:
        return f"[S{self.rank + 1}]"

    @property
    def support(self) -> float:
        """Explainable support signal used by the refusal guardrail."""
        return round(0.5 * self.similarity + 0.5 * self.query_coverage, 4)


@dataclass
class Citation:
    label: str  # "[S1]"
    chunk_id: str
    doc_id: str
    doc_title: str
    heading: str
    quote: str  # the sentence(s) of the source that support the answer
    similarity: float = 0.0


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    reason: str = ""
    support: float = 0.0  # fraction of answer sentences backed by their citations
    unsupported_sentences: list[str] = field(default_factory=list)
    retrieved: list[Retrieved] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["retrieved"] = [
            {
                "label": r.label,
                "chunk_id": r.chunk.chunk_id,
                "doc_id": r.chunk.doc_id,
                "heading": r.chunk.heading,
                "score": round(r.score, 6),
                "similarity": round(r.similarity, 4),
                "query_coverage": round(r.query_coverage, 4),
                "sources": list(r.sources),
                "text": r.chunk.text,
            }
            for r in self.retrieved
        ]
        return data
