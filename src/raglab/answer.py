"""The grounded answerer: retrieval, then three guardrails, then an answer.

Guardrails (each one is a test case in ``tests/test_answer.py``):

G1  low_support        - if the best retrieved passage does not sufficiently
                         cover the question, refuse instead of guessing.
G2  no_valid_citation  - every [S#] in the draft must exist in the context;
                         unknown labels are dropped. If nothing valid remains,
                         the model gets one stricter retry, then we refuse.
G3  unsupported_sentence - each sentence must be supported by the source it
                         cites (term coverage); with ``strict=True`` the
                         unsupported sentences are removed from the answer.

The output is an ``Answer`` with the citations, the refusal reason, the support
score, and a full trace - so a human can audit *why* the system said what it said.
"""

from __future__ import annotations

import re
import time

from .llm import NOT_FOUND, LLMClient, Source
from .retrieval import HybridIndex
from .text import content_tokens, coverage, split_sentences
from .types import Answer, Citation, Retrieved

_CITATION_RE = re.compile(r"\[S(\d+)\]")


class GroundedAnswerer:
    def __init__(
        self,
        index: HybridIndex,
        llm: LLMClient,
        k: int = 5,
        max_context: int = 4,
        mode: str = "hybrid",
        min_support: float = 0.35,
        sentence_support: float = 0.6,
        require_citations: bool = True,
        strict: bool = False,
    ) -> None:
        self.index = index
        self.llm = llm
        self.k = k
        self.max_context = max_context
        self.mode = mode
        self.min_support = min_support
        self.sentence_support = sentence_support
        self.require_citations = require_citations
        self.strict = strict

    # -- public API ---------------------------------------------------------
    def answer(self, question: str) -> Answer:
        started = time.perf_counter()
        retrieved = self.index.search(question, k=self.k, mode=self.mode)
        context = retrieved[: self.max_context]
        trace: dict = {
            "llm": getattr(self.llm, "name", self.llm.__class__.__name__),
            "mode": self.mode,
            "min_support": self.min_support,
            "guardrails": [],
            "retrieved": [
                {"label": r.label, "chunk_id": r.chunk.chunk_id, "support": r.support} for r in retrieved
            ],
        }

        answer = Answer(question=question, text="", retrieved=retrieved, trace=trace)
        if not retrieved:
            return self._refuse(answer, "no_documents_indexed", trace, started)

        # --- G1: is there enough evidence in the corpus at all? ------------
        best = max(retrieved, key=lambda r: r.support)
        trace["best_support"] = best.support
        if best.support < self.min_support:
            trace["guardrails"].append(
                {"rule": "low_support", "action": "refuse", "detail": {"support": best.support}}
            )
            return self._refuse(answer, "low_retrieval_support", trace, started)
        trace["guardrails"].append(
            {"rule": "low_support", "action": "pass", "detail": {"support": best.support}}
        )

        sources = [self._as_source(r) for r in context]
        raw = (self.llm.generate(question, sources) or "").strip()
        trace["raw_output"] = raw

        if not raw or NOT_FOUND.lower() in raw.lower():
            trace["guardrails"].append({"rule": "model_refusal", "action": "refuse", "detail": {"raw": raw}})
            return self._refuse(answer, "insufficient_context_in_sources", trace, started)

        citations, unknown = self._collect_citations(raw, context)
        trace["guardrails"].append(
            {
                "rule": "no_valid_citation",
                "action": "pass" if citations else "retry",
                "detail": {"valid": [c.label for c in citations], "dropped": unknown},
            }
        )

        # --- G2: citation contract ----------------------------------------
        if not citations and self.require_citations:
            raw_retry = (self.llm.generate(question, sources, strict=True) or "").strip()
            trace["raw_output_retry"] = raw_retry
            if not raw_retry or NOT_FOUND.lower() in raw_retry.lower():
                return self._refuse(answer, "insufficient_context_in_sources", trace, started)
            citations, unknown = self._collect_citations(raw_retry, context)
            raw = raw_retry
            if not citations:
                trace["guardrails"].append({"rule": "no_valid_citation", "action": "refuse", "detail": {}})
                return self._refuse(answer, "no_valid_citation", trace, started)

        # --- G3: sentence-level support ------------------------------------
        evaluated = self._evaluate_sentences(raw, context)
        supported = [sentence for sentence, ok in evaluated if ok]
        unsupported = [sentence for sentence, ok in evaluated if not ok]
        trace["guardrails"].append(
            {
                "rule": "unsupported_sentence",
                "action": "strip" if (unsupported and self.strict) else ("flag" if unsupported else "pass"),
                "detail": {"unsupported": unsupported},
            }
        )
        if not supported:
            return self._refuse(answer, "no_supported_content", trace, started)

        text = " ".join(supported) if self.strict else " ".join(sentence for sentence, _ in evaluated)
        support = round(len(supported) / len(evaluated), 4) if evaluated else 0.0
        answer.text = text
        answer.citations = self._citations_for(text, citations, context)
        answer.support = support
        answer.unsupported_sentences = unsupported
        answer.trace = trace
        answer.trace["timings_ms"] = {"total": round((time.perf_counter() - started) * 1000, 2)}
        return answer

    # -- internals ----------------------------------------------------------
    @staticmethod
    def _as_source(item: Retrieved) -> Source:
        return Source(
            label=item.label,
            doc_id=item.chunk.doc_id,
            heading=item.chunk.heading,
            text=item.chunk.text,
        )

    @staticmethod
    def _refuse(answer: Answer, reason: str, trace: dict, started: float) -> Answer:
        answer.text = ""
        answer.refused = True
        answer.reason = reason
        answer.citations = []
        trace["timings_ms"] = {"total": round((time.perf_counter() - started) * 1000, 2)}
        answer.trace = trace
        return answer

    def _collect_citations(self, text: str, context: list[Retrieved]) -> tuple[list[Citation], list[str]]:
        by_label = {item.label: item for item in context}
        seen: list[Citation] = []
        dropped: list[str] = []
        for match in _CITATION_RE.finditer(text):
            label = f"[S{match.group(1)}]"
            item = by_label.get(label)
            if item is None:
                if label not in dropped:
                    dropped.append(label)
                continue
            if any(c.label == label for c in seen):
                continue
            seen.append(
                Citation(
                    label=label,
                    chunk_id=item.chunk.chunk_id,
                    doc_id=item.chunk.doc_id,
                    doc_title=item.chunk.doc_title,
                    heading=item.chunk.heading,
                    quote="",
                    similarity=item.similarity,
                )
            )
        return seen, dropped

    def _evaluate_sentences(self, text: str, context: list[Retrieved]) -> list[tuple[str, bool]]:
        """Return every answer sentence with a flag saying whether its citation backs it."""
        if not self.require_citations:
            return [(_ensure_period(text.strip()), True)] if text.strip() else []
        by_label = {item.label: item for item in context}
        all_text = " ".join(item.chunk.text for item in context)
        evaluated: list[tuple[str, bool]] = []
        for sentence in split_sentences(text):
            labels = [f"[S{m.group(1)}]" for m in _CITATION_RE.finditer(sentence)]
            support_text = (
                " ".join(by_label[label].chunk.text for label in labels if label in by_label) or all_text
            )
            tokens = content_tokens(_CITATION_RE.sub(" ", sentence))
            if not tokens:
                continue
            score = coverage(tokens, support_text)
            evaluated.append((_ensure_period(sentence.strip()), score >= self.sentence_support))
        return evaluated

    @staticmethod
    def _citations_for(text: str, citations: list[Citation], context: list[Retrieved]) -> list[Citation]:
        by_chunk = {item.chunk.chunk_id: item for item in context}
        enriched: list[Citation] = []
        for citation in citations:
            item = by_chunk.get(citation.chunk_id)
            quote = ""
            if item is not None:
                quote = _best_quote(text, item.chunk.text)
            enriched.append(
                Citation(
                    label=citation.label,
                    chunk_id=citation.chunk_id,
                    doc_id=citation.doc_id,
                    doc_title=citation.doc_title,
                    heading=citation.heading,
                    quote=quote,
                    similarity=citation.similarity,
                )
            )
        return enriched


def _ensure_period(sentence: str) -> str:
    return sentence if sentence.endswith((".", "!", "?", "。", "！", "？", ":")) else sentence + "."


def _best_quote(answer_text: str, source_text: str) -> str:
    """The sentence of the source that best supports the answer (for display)."""
    answer_tokens = content_tokens(answer_text)
    best, best_score = "", 0.0
    for sentence in split_sentences(source_text):
        score = coverage(content_tokens(sentence), answer_text) if answer_tokens else 0.0
        if score > best_score:
            best, best_score = sentence, score
    return best or (split_sentences(source_text) or [""])[0]
