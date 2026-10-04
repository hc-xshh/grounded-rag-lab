"""Okapi BM25 - the lexical half of the hybrid retriever (pure standard library)."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


class BM25Okapi:
    """Classic BM25 with smoothed IDF.

    Lexical matching still wins on exact identifiers (error codes, plan names,
    field names) which is exactly where embedding-only retrieval fails on
    business documents.
    """

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = [list(doc) for doc in corpus]
        self.doc_len = [len(doc) for doc in self.corpus]
        self.avg_len = (sum(self.doc_len) / len(self.corpus)) if self.corpus else 0.0
        self.frequencies = [Counter(doc) for doc in self.corpus]
        df: Counter[str] = Counter()
        for doc in self.corpus:
            df.update(set(doc))
        n_docs = len(self.corpus)
        self.idf = {term: math.log(1 + (n_docs - count + 0.5) / (count + 0.5)) for term, count in df.items()}

    def get_scores(self, query: Sequence[str]) -> list[float]:
        scores = [0.0] * len(self.corpus)
        if not self.corpus:
            return scores
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freq in enumerate(self.frequencies):
                tf = freq.get(term)
                if not tf:
                    continue
                norm = self.doc_len[i] / self.avg_len if self.avg_len else 1.0
                scores[i] += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * norm))
        return scores
