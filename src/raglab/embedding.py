"""Pluggable embedding: an offline TF-IDF vectoriser by default, any
OpenAI-compatible /embeddings endpoint when a key is present.

The interface is intentionally tiny (fit + embed -> sparse {index: weight}) so a
neural embedder can be dropped in without touching retrieval or evaluation.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from .text import tokenize


class Embedder(Protocol):
    name: str

    def fit(self, texts: Sequence[str]) -> None: ...

    def embed(self, text: str) -> dict[int, float]: ...

    def embed_many(self, texts: Sequence[str]) -> list[dict[int, float]]: ...


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    """Cosine similarity between two sparse vectors (both are L2-normalised)."""
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = 0.0
    for idx, weight in a.items():
        other = b.get(idx)
        if other is not None:
            dot += weight * other
    return dot


class TfidfEmbedder:
    """Deterministic, dependency-free TF-IDF embedder.

    It is not a neural model, and the README says so. What it gives you is a
    reproducible baseline that runs anywhere: perfect for tests, CI, and for
    letting a client try the project without an API key.
    """

    name = "tfidf-offline"

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}
        self.idf: list[float] = []
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> None:
        df: Counter[str] = Counter()
        for text in texts:
            df.update(set(tokenize(text)))
        self.vocabulary = {term: i for i, term in enumerate(sorted(df))}
        n_docs = max(len(texts), 1)
        self.idf = [0.0] * len(self.vocabulary)
        for term, count in df.items():
            # smoothed inverse document frequency
            self.idf[self.vocabulary[term]] = math.log((1 + n_docs) / (1 + count)) + 1.0
        self._fitted = True

    def embed(self, text: str) -> dict[int, float]:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit() must be called before embed()")
        counts = Counter(tokenize(text))
        vector: dict[int, float] = {}
        for term, count in counts.items():
            idx = self.vocabulary.get(term)
            if idx is None:
                continue  # out-of-vocabulary terms are handled by BM25
            vector[idx] = (1.0 + math.log(count)) * self.idf[idx]
        norm = math.sqrt(sum(v * v for v in vector.values()))
        if norm:
            for idx in vector:
                vector[idx] /= norm
        return vector

    def embed_many(self, texts: Sequence[str]) -> list[dict[int, float]]:
        return [self.embed(t) for t in texts]

    # -- persistence: lets a fitted index be cached to disk ------------------
    def to_dict(self) -> dict:
        return {"name": self.name, "vocabulary": self.vocabulary, "idf": self.idf}

    @classmethod
    def from_dict(cls, data: dict) -> TfidfEmbedder:
        embedder = cls()
        embedder.vocabulary = {k: int(v) for k, v in data["vocabulary"].items()}
        embedder.idf = [float(x) for x in data["idf"]]
        embedder._fitted = True
        return embedder


class OpenAIEmbedder:
    """Thin wrapper over any OpenAI-compatible /v1/embeddings endpoint.

    Enabled only when OPENAI_API_KEY is set; falls back to TF-IDF otherwise.
    """

    name = "openai-embeddings"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.model = model or os.environ.get("RAGLAB_EMBED_MODEL", "text-embedding-3-small")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip(
            "/"
        )
        self.timeout = timeout
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> None:  # nothing to learn
        self._fitted = True

    def embed(self, text: str) -> dict[int, float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[dict[int, float]]:
        payload = json.dumps({"model": self.model, "input": list(texts)}).encode()
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode())
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise RuntimeError(f"embedding request failed: {exc}") from exc
        vectors: list[dict[int, float]] = []
        for item in body["data"]:
            dense = item["embedding"]
            norm = math.sqrt(sum(v * v for v in dense)) or 1.0
            vectors.append({i: v / norm for i, v in enumerate(dense) if v})
        return vectors


def default_embedder(prefer_remote: bool = False) -> Embedder:
    """Pick an embedder: remote when explicitly asked for and keyed, else offline."""
    if prefer_remote and os.environ.get("OPENAI_API_KEY"):
        return OpenAIEmbedder()
    return TfidfEmbedder()
