from pathlib import Path

from raglab.evaluate import GoldenCase, evaluate, load_golden, render_markdown
from raglab.pipeline import RagPipeline

ROOT = Path(__file__).resolve().parents[1]


def test_metrics_are_computed_on_a_tiny_set():
    pipeline = RagPipeline.from_dir(ROOT / "data" / "docs", k=5)
    cases = [
        GoldenCase(
            id="hit",
            question="What is the API rate limit on the Growth plan?",
            answerable=True,
            expected_doc_ids=["rate-limits-and-errors"],
            expected_facts=["300 requests per minute"],
        ),
        GoldenCase(
            id="gap",
            question="Is there a mobile app for iOS and Android?",
            answerable=False,
        ),
    ]
    report = evaluate(pipeline, cases, k=5)
    m = report.metrics
    assert m["cases"] == 2
    assert m["retrieval_recall_at_k"] == 1.0
    assert m["mrr"] == 1.0
    assert m["fact_coverage"] == 1.0
    assert m["refusal_accuracy"] == 1.0
    assert m["invalid_citations"] == 0
    assert m["spurious_refusals"] == 0
    assert m["answer_support"] == 1.0


def test_missing_fact_lowers_coverage():
    pipeline = RagPipeline.from_dir(ROOT / "data" / "docs", k=5)
    cases = [
        GoldenCase(
            id="made-up-fact",
            question="What is the API rate limit on the Growth plan?",
            answerable=True,
            expected_doc_ids=["rate-limits-and-errors"],
            expected_facts=["300 requests per minute", "9,999 requests per minute"],
        )
    ]
    report = evaluate(pipeline, cases, k=5)
    assert report.metrics["fact_coverage"] == 0.5
    assert report.cases[0].facts_missing == ["9,999 requests per minute"]


def test_render_markdown_and_jsonl(tmp_path: Path):
    pipeline = RagPipeline.from_dir(ROOT / "data" / "docs", k=5)
    cases = load_golden(ROOT / "data" / "eval" / "golden.jsonl")
    report = evaluate(pipeline, cases, k=5)
    assert len(report.cases) == len(cases)

    markdown = render_markdown(report)
    assert markdown.startswith("# Evaluation report")
    assert "| Retrieval recall@k |" in markdown
    assert "## Per-case results" in markdown

    target = tmp_path / "results.jsonl"
    report.to_jsonl(target)
    lines = target.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(cases)


def test_golden_set_covers_both_directions():
    cases = load_golden(ROOT / "data" / "eval" / "golden.jsonl")
    assert any(c.answerable for c in cases)
    assert any(not c.answerable for c in cases)
    assert len({c.id for c in cases}) == len(cases)
    assert all(c.question.endswith("?") for c in cases)
