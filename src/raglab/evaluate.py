"""Offline evaluation harness.

A RAG system without measurement is a demo. This module runs a golden set of
questions - answerable ones with the document(s) and facts that must appear in
the answer, plus unanswerable ones that *must* be refused - and reports:

* retrieval_recall@k / MRR        - did we even fetch the right passage?
* fact_coverage                   - is the expected fact in the answer?
* citation_precision              - do the citations point at the right documents?
* invalid_citations               - hallucinated labels that survived (must stay 0)
* refusal_accuracy                - answered what is answerable, refused what is not
* answer_support                  - share of sentences backed by their citation
* latency p50/p95                 - honest timing, not a marketing number

Results are written as markdown (human), JSONL (machine) and optionally an
interactive HTML page.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .pipeline import RagPipeline
from .text import normalise_fact
from .types import Answer


@dataclass
class GoldenCase:
    id: str
    question: str
    answerable: bool
    expected_doc_ids: list[str] = field(default_factory=list)
    expected_facts: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class CaseResult:
    id: str
    question: str
    answerable: bool
    refused: bool
    reason: str
    answer: str
    citations: list[str]
    citation_docs: list[str]
    retrieved_docs: list[str]
    recall_hit: bool
    reciprocal_rank: float
    facts_found: list[str]
    facts_missing: list[str]
    support: float
    latency_ms: float
    best_support: float = 0.0
    unanswerable_correct: bool | None = None
    invalid_citations: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    metrics: dict[str, Any]
    cases: list[CaseResult]
    meta: dict[str, Any]

    def to_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for case in self.cases:
                handle.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")


def load_golden(path: Path) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        raw = json.loads(line)
        cases.append(
            GoldenCase(
                id=raw["id"],
                question=raw["question"],
                answerable=bool(raw.get("answerable", True)),
                expected_doc_ids=list(raw.get("expected_doc_ids", [])),
                expected_facts=list(raw.get("expected_facts", [])),
                note=raw.get("note", ""),
            )
        )
    return cases


def _facts_found(answer_text: str, facts: Sequence[str]) -> tuple[list[str], list[str]]:
    haystack = normalise_fact(answer_text)
    found, missing = [], []
    for fact in facts:
        needle = normalise_fact(fact)
        (found if needle and needle in haystack else missing).append(fact)
    return found, missing


def evaluate_case(pipeline: RagPipeline, case: GoldenCase, k: int = 5) -> CaseResult:
    answer: Answer = pipeline.ask(case.question, k=k)
    retrieved_docs = [r.chunk.doc_id for r in answer.retrieved]
    expected = set(case.expected_doc_ids)
    recall_hit = bool(expected & set(retrieved_docs)) if expected else False
    reciprocal_rank = 0.0
    for position, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in expected:
            reciprocal_rank = 1.0 / position
            break
    found, missing = _facts_found(answer.text, case.expected_facts)
    invalid = [
        c.label for c in answer.citations if c.chunk_id not in {r.chunk.chunk_id for r in answer.retrieved}
    ]
    return CaseResult(
        id=case.id,
        question=case.question,
        answerable=case.answerable,
        refused=answer.refused,
        reason=answer.reason,
        answer=answer.text,
        citations=[c.label for c in answer.citations],
        citation_docs=[c.doc_id for c in answer.citations],
        retrieved_docs=retrieved_docs,
        recall_hit=recall_hit,
        reciprocal_rank=round(reciprocal_rank, 4),
        facts_found=found,
        facts_missing=missing,
        support=answer.support,
        latency_ms=float(answer.trace.get("timings_ms", {}).get("total", 0.0)),
        best_support=float(answer.trace.get("best_support", 0.0)),
        unanswerable_correct=(answer.refused if not case.answerable else None),
        invalid_citations=invalid,
    )


def _mean(values: Sequence[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return round(ordered[index], 2)


def summarise(results: Sequence[CaseResult], meta: dict[str, Any]) -> EvalReport:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    answered = [r for r in results if not r.refused]

    with_facts = [r for r in answerable if r.facts_found or r.facts_missing]
    total_facts = sum(len(r.facts_found) + len(r.facts_missing) for r in with_facts)

    cited_docs_total = sum(len(r.citation_docs) for r in answered)

    # citation precision needs the expected docs per case, carried on the case
    precision_hits = 0
    precision_total = 0
    for result in answered:
        expected = set(meta.get("expected_docs_by_case", {}).get(result.id, []))
        for doc in result.citation_docs:
            precision_total += 1
            if doc in expected:
                precision_hits += 1

    refusals_correct = sum(1 for r in results if (r.refused if not r.answerable else not r.refused))
    metrics = {
        "cases": len(results),
        "answerable_cases": len(answerable),
        "unanswerable_cases": len(unanswerable),
        "retrieval_recall_at_k": _mean([1.0 if r.recall_hit else 0.0 for r in answerable]),
        "mrr": _mean([r.reciprocal_rank for r in answerable]),
        "fact_coverage": round(sum(len(r.facts_found) for r in with_facts) / total_facts, 4)
        if total_facts
        else 0.0,
        "citation_precision": round(precision_hits / precision_total, 4) if precision_total else 0.0,
        "invalid_citations": sum(len(r.invalid_citations) for r in results),
        "refusal_accuracy": round(refusals_correct / len(results), 4) if results else 0.0,
        "refusal_rate": round((len(results) - len(answered)) / len(results), 4) if results else 0.0,
        "spurious_refusals": sum(1 for r in answerable if r.refused),
        "answer_support": _mean([r.support for r in answered]),
        "latency_p50_ms": _percentile([r.latency_ms for r in results], 50),
        "latency_p95_ms": _percentile([r.latency_ms for r in results], 95),
        "cited_passages": cited_docs_total,
    }
    return EvalReport(metrics=metrics, cases=list(results), meta=meta)


def evaluate(pipeline: RagPipeline, cases: Sequence[GoldenCase], k: int = 5) -> EvalReport:
    results = [evaluate_case(pipeline, case, k=k) for case in cases]
    meta = {
        "k": k,
        "corpus": pipeline.stats().as_row(),
        "expected_docs_by_case": {c.id: list(c.expected_doc_ids) for c in cases},
        "answerer": {
            "mode": pipeline.answerer.mode,
            "min_support": pipeline.answerer.min_support,
            "sentence_support": pipeline.answerer.sentence_support,
            "max_context": pipeline.answerer.max_context,
        },
    }
    return summarise(results, meta)


# -- rendering -------------------------------------------------------------
METRIC_LABELS = [
    (
        "retrieval_recall_at_k",
        "Retrieval recall@k",
        "share of answerable questions whose source document was retrieved",
    ),
    ("mrr", "MRR", "mean reciprocal rank of the first correct document"),
    ("fact_coverage", "Fact coverage", "expected facts present in the answers"),
    ("citation_precision", "Citation precision", "citations that point at the expected document"),
    (
        "invalid_citations",
        "Invalid citations",
        "labels that did not exist in the context (guardrail target: 0)",
    ),
    ("refusal_accuracy", "Refusal accuracy", "answered what is answerable AND refused what is not"),
    (
        "spurious_refusals",
        "Spurious refusals",
        "answerable questions that were refused anyway (lower is better)",
    ),
    ("answer_support", "Answer support", "share of answer sentences backed by their citation"),
    ("latency_p50_ms", "Latency p50", "median end-to-end answer latency in ms"),
    ("latency_p95_ms", "Latency p95", "95th percentile latency in ms"),
]


def render_markdown(report: EvalReport) -> str:
    m = report.metrics
    corpus = report.meta.get("corpus", {})
    lines: list[str] = []
    lines.append("# Evaluation report")
    lines.append("")
    lines.append(
        f"Corpus: **{corpus.get('documents', '?')} documents / {corpus.get('chunks', '?')} chunks** · "
        f"embedder: `{corpus.get('embedder', '?')}` · answerer: `{corpus.get('llm', '?')}` · "
        f"top-k: **{report.meta.get('k', '?')}**"
    )
    lines.append("")
    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| metric | value | what it means |")
    lines.append("| --- | --- | --- |")
    for key, label, meaning in METRIC_LABELS:
        value = m.get(key, "n/a")
        lines.append(f"| {label} | {value} | {meaning} |")
    lines.append("")
    lines.append("## Per-case results")
    lines.append("")
    lines.append("| # | question | answerable | outcome | recall | support | citations |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for i, case in enumerate(report.cases, start=1):
        outcome = "refused" if case.refused else "answered"
        if not case.answerable:
            outcome += " (correct)" if case.refused else " **WRONG**"
        elif case.refused:
            outcome += " **spurious**"
        lines.append(
            f"| {i} | {case.question} | {'yes' if case.answerable else 'no'} | {outcome} | "
            f"{'hit' if case.recall_hit else ('miss' if case.answerable else 'n/a')} | "
            f"{case.support:.2f} | {' '.join(case.citations) or '-'} |"
        )
    lines.append("")
    lines.append("## Refusals")
    lines.append("")
    refused = [c for c in report.cases if c.refused]
    if not refused:
        lines.append("_none_")
    else:
        for case in refused:
            kind = "expected" if not case.answerable else "SPURIOUS"
            lines.append(f"- `{case.id}` ({kind}) - {case.question} → `{case.reason}`")
    lines.append("")
    lines.append("## Reproducing this report")
    lines.append("")
    lines.append("```bash")
    lines.append("make install")
    lines.append("make eval          # writes reports/ + docs/index.html")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)
