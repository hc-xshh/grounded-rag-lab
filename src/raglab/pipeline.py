"""Pipeline wiring: documents in, grounded answers out."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .answer import GroundedAnswerer
from .chunking import chunk_markdown
from .embedding import Embedder, default_embedder
from .llm import LLMClient, default_llm
from .retrieval import HybridIndex
from .types import Answer, Chunk

_TITLE_RE = re.compile(r"^#\s+(.*)$", re.M)


@dataclass
class CorpusStats:
    documents: int
    chunks: int
    avg_chunk_chars: int
    embedder: str
    llm: str

    def as_row(self) -> dict:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "avg_chunk_chars": self.avg_chunk_chars,
            "embedder": self.embedder,
            "llm": self.llm,
        }


def load_documents(paths: Iterable[Path]) -> list[Chunk]:
    """Read markdown files into chunks, keyed by their relative path."""
    chunks: list[Chunk] = []
    for path in sorted(paths):
        text = path.read_text(encoding="utf-8")
        doc_id = path.stem
        title_match = _TITLE_RE.search(text)
        title = title_match.group(1).strip() if title_match else doc_id
        chunks.extend(chunk_markdown(text, doc_id=doc_id, doc_title=title))
    return chunks


class RagPipeline:
    """Small, explicit object that owns the index and the answerer."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
        prefer_remote: bool = False,
        **answerer_kwargs,
    ) -> None:
        self.chunks = list(chunks)
        self.llm = llm or default_llm(prefer_remote=prefer_remote)
        self.embedder = embedder or default_embedder(prefer_remote=prefer_remote)
        self.index = HybridIndex(self.chunks, embedder=self.embedder)
        self.answerer = GroundedAnswerer(self.index, self.llm, **answerer_kwargs)

    @classmethod
    def from_dir(cls, docs_dir: Path, pattern: str = "**/*.md", **kwargs) -> RagPipeline:
        paths = sorted(Path(docs_dir).glob(pattern))
        if not paths:
            raise FileNotFoundError(f"no documents matching {pattern!r} under {docs_dir}")
        return cls(load_documents(paths), **kwargs)

    @classmethod
    def from_paths(cls, paths: Iterable[Path], **kwargs) -> RagPipeline:
        return cls(load_documents(paths), **kwargs)

    def ask(self, question: str, **kwargs) -> Answer:
        answerer = self.answerer
        if kwargs:
            answerer = GroundedAnswerer(self.index, self.llm, **{**self._answerer_kwargs(), **kwargs})
        return answerer.answer(question)

    def _answerer_kwargs(self) -> dict:
        a = self.answerer
        return {
            "k": a.k,
            "max_context": a.max_context,
            "mode": a.mode,
            "min_support": a.min_support,
            "sentence_support": a.sentence_support,
            "require_citations": a.require_citations,
            "strict": a.strict,
        }

    def stats(self) -> CorpusStats:
        sizes = [len(c.text) for c in self.chunks] or [0]
        return CorpusStats(
            documents=len({c.doc_id for c in self.chunks}),
            chunks=len(self.chunks),
            avg_chunk_chars=round(sum(sizes) / len(sizes)),
            embedder=getattr(self.embedder, "name", self.embedder.__class__.__name__),
            llm=getattr(self.llm, "name", self.llm.__class__.__name__),
        )
