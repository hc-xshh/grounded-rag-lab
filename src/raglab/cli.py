"""Command line interface: ingest, ask, eval (with HTML report)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluate import evaluate, load_golden, render_markdown
from .pipeline import RagPipeline
from .report import write_html

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS = ROOT / "data" / "docs"
DEFAULT_GOLDEN = ROOT / "data" / "eval" / "golden.jsonl"
DEFAULT_REPORTS = ROOT / "reports"
DEFAULT_HTML = ROOT / "docs" / "index.html"

BAR = "─" * 72


def _pipeline(args: argparse.Namespace) -> RagPipeline:
    return RagPipeline.from_dir(
        Path(args.docs),
        prefer_remote=args.remote,
        k=args.k,
        min_support=args.min_support,
        strict=args.strict,
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args)
    stats = pipeline.stats()
    if args.json:
        print(json.dumps(stats.as_row(), indent=2))
        return 0
    print(f"documents : {stats.documents}")
    print(f"chunks    : {stats.chunks}  (avg {stats.avg_chunk_chars} chars)")
    print(f"embedder  : {stats.embedder}")
    print(f"answerer  : {stats.llm}")
    print()
    print("sample chunks")
    for chunk in pipeline.chunks[:5]:
        print(f"  · {chunk.chunk_id:<22} {chunk.heading[:52]:<52} {len(chunk.text):>4} chars")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args)
    answer = pipeline.ask(args.question)
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2, ensure_ascii=False))
        return 0
    print(BAR)
    print(f"Q: {answer.question}")
    print(BAR)
    if answer.refused:
        print(f"REFUSED  (reason: {answer.reason})")
        print("The sources do not support an answer, so the system declines instead of guessing.")
    else:
        print(answer.text)
        print()
        print(f"support={answer.support:.2f}  citations={len(answer.citations)}")
        for citation in answer.citations:
            print(f"  {citation.label} {citation.doc_id} · {citation.heading}")
            if citation.quote:
                print(f'      "{citation.quote}"')
    print()
    print("retrieved")
    for item in answer.retrieved:
        print(
            f"  {item.label} sim={item.similarity:.3f} cov={item.query_coverage:.2f} "
            f"support={item.support:.3f}  {item.chunk.doc_id} · {item.chunk.heading[:40]}"
        )
    print()
    print("guardrails")
    for step in answer.trace.get("guardrails", []):
        print(
            f"  {step['rule']:<22} {step['action']:<7} {json.dumps(step['detail'], ensure_ascii=False)[:90]}"
        )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    pipeline = _pipeline(args)
    cases = load_golden(Path(args.golden))
    report = evaluate(pipeline, cases, k=args.k)

    reports_dir = Path(args.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = reports_dir / "eval-report.md"
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    report.to_jsonl(reports_dir / "eval-results.jsonl")

    if not args.no_html:
        demo = []
        for case in cases[: args.demo_size]:
            answer = pipeline.ask(case.question)
            expected = ", ".join(case.expected_doc_ids) or "no source exists (must refuse)"
            kind = "expect answer" if case.answerable else "expect refusal"
            demo.append((answer, expected, kind))
        write_html(Path(args.html), demo, report)

    if args.json:
        print(json.dumps(report.metrics, indent=2))
    else:
        print(BAR)
        print("evaluation summary")
        print(BAR)
        for key, value in report.metrics.items():
            print(f"  {key:<24} {value}")
        print()
        print(f"markdown : {markdown_path}")
        print(f"jsonl    : {reports_dir / 'eval-results.jsonl'}")
        if not args.no_html:
            print(f"html     : {args.html}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raglab",
        description="Citation-first RAG with guardrails and an offline evaluation harness.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--docs", default=str(DEFAULT_DOCS), help="directory with markdown documents")
        p.add_argument("--k", type=int, default=5, help="passages retrieved per question")
        p.add_argument(
            "--min-support",
            type=float,
            default=0.35,
            help="refuse when the best passage supports the question less than this",
        )
        p.add_argument("--strict", action="store_true", help="strip unsupported sentences from answers")
        p.add_argument(
            "--remote",
            action="store_true",
            help="use OPENAI_API_KEY + RAGLAB_LLM_MODEL instead of the offline answerer",
        )

    ingest = sub.add_parser("ingest", help="chunk and index the corpus")
    common(ingest)
    ingest.add_argument("--json", action="store_true")
    ingest.set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask", help="ask a single question")
    common(ask)
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true")
    ask.set_defaults(func=cmd_ask)

    ev = sub.add_parser("eval", help="run the golden set and write reports")
    common(ev)
    ev.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    ev.add_argument("--reports", default=str(DEFAULT_REPORTS))
    ev.add_argument("--html", default=str(DEFAULT_HTML))
    ev.add_argument("--demo-size", type=int, default=10, help="questions replayed in the HTML page")
    ev.add_argument("--no-html", action="store_true")
    ev.add_argument("--json", action="store_true")
    ev.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
