"""Hybrid retrieval: BM25 + vector search fused with Reciprocal Rank Fusion.

Why hybrid: on business documents the two retrievers fail differently. Vector
search misses exact identifiers, BM25 misses paraphrase. Fusing their *ranks*
(not their scores) avoids having to normalise two incomparable scales.
"""

from __future__ import annotations

from collections.abc import Sequence

from .bm25 import BM25Okapi
from .embedding import Embedder, TfidfEmbedder, cosine
from .text import content_tokens, coverage, tokenize
from .types import Chunk, Retrieved

MODES = ("hybrid", "vector", "bm25")


class HybridIndex:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder | None = None,
        rrf_k: int = 60,
        candidate_pool: int = 50,
    ) -> None:
        self.chunks: list[Chunk] = list(chunks)
        self.embedder: Embedder = embedder or TfidfEmbedder()
        self.rrf_k = rrf_k
        self.candidate_pool = min(candidate_pool, max(len(self.chunks), 1))
        self.embedder.fit([c.indexed_text for c in self.chunks])
        self.vectors = self.embedder.embed_many([c.indexed_text for c in self.chunks])
        self.bm25 = BM25Okapi([tokenize(c.indexed_text) for c in self.chunks])

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Retrieved]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if not self.chunks or not query.strip():
            return []

        pool = self.candidate_pool
        query_tokens = tokenize(query)
        query_content = content_tokens(query)

        vector_sims = [cosine(self.embedder.embed(query), v) for v in self.vectors]
        bm25_scores = self.bm25.get_scores(query_tokens)

        # Stable sorts keep ties in document order -> deterministic output.
        vector_ranked = sorted(range(len(self.chunks)), key=lambda i: (-vector_sims[i], i))[:pool]
        bm25_ranked = sorted(range(len(self.chunks)), key=lambda i: (-bm25_scores[i], i))[:pool]

        fused: dict[int, float] = {}
        sources: dict[int, list[str]] = {}
        ranked_lists: list[tuple[str, list[int]]] = []
        if mode in ("vector", "hybrid"):
            ranked_lists.append(("vector", vector_ranked))
        if mode in ("bm25", "hybrid"):
            ranked_lists.append(("bm25", bm25_ranked))

        for name, ranked in ranked_lists:
            for rank, idx in enumerate(ranked, start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)
                sources.setdefault(idx, []).append(name)

        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        results: list[Retrieved] = []
        for rank, (idx, score) in enumerate(ordered):
            chunk = self.chunks[idx]
            results.append(
                Retrieved(
                    chunk=chunk,
                    rank=rank,
                    score=score,
                    similarity=round(vector_sims[idx], 6),
                    query_coverage=coverage(query_content, chunk.indexed_text),
                    sources=tuple(sorted(sources.get(idx, []))),
                )
            )
        return results
