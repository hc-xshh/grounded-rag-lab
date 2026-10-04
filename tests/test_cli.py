"""End-to-end CLI tests - the commands a reviewer will actually run."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "raglab", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_ingest_reports_corpus_stats():
    result = run("ingest", "--json")
    assert result.returncode == 0, result.stderr
    stats = json.loads(result.stdout)
    assert stats["documents"] == 5
    assert stats["chunks"] > 20
    assert stats["embedder"] == "tfidf-offline"


def test_ask_prints_answer_with_citations():
    result = run("ask", "What is the API rate limit on the Growth plan?")
    assert result.returncode == 0, result.stderr
    assert "300 requests per minute" in result.stdout
    assert "citations=" in result.stdout
    assert "[S1]" in result.stdout
    assert "guardrails" in result.stdout


def test_ask_refuses_an_unanswerable_question():
    result = run("ask", "Is there a mobile app for iOS and Android?")
    assert result.returncode == 0, result.stderr
    assert "REFUSED" in result.stdout
    assert "low_retrieval_support" in result.stdout


def test_eval_writes_reports(tmp_path: Path):
    reports = tmp_path / "reports"
    result = run("eval", "--reports", str(reports), "--no-html")
    assert result.returncode == 0, result.stderr
    assert "refusal_accuracy" in result.stdout
    assert (reports / "eval-report.md").exists()
    assert (reports / "eval-results.jsonl").exists()
    rows = (reports / "eval-results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 34


def test_eval_writes_the_html_report(tmp_path: Path):
    html_path = tmp_path / "index.html"
    result = run(
        "eval",
        "--reports",
        str(tmp_path / "reports"),
        "--html",
        str(html_path),
        "--demo-size",
        "3",
    )
    assert result.returncode == 0, result.stderr
    html = html_path.read_text(encoding="utf-8")
    assert "<!doctype html>" in html
    assert "__RAGLAB_DATA__" not in html  # the payload must be embedded, not templated
    assert "grounded-rag-lab" in html
    assert "Refusal accuracy" in html  # headline metrics are rendered into the page


def test_bad_docs_directory_exits_with_error(tmp_path: Path):
    result = run("ingest", "--docs", str(tmp_path / "nope"))
    assert result.returncode == 2
    assert "error:" in result.stderr
