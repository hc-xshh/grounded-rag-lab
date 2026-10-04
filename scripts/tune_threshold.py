#!/usr/bin/env python3
"""Sweep min_support and report how the two populations separate.

Answerable questions should pass the gate, unanswerable ones should not. The
numbers here are what justify the default threshold in the CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raglab.evaluate import evaluate, load_golden  # noqa: E402
from raglab.pipeline import RagPipeline  # noqa: E402

root = Path(__file__).resolve().parents[1]
cases = load_golden(root / "data/eval/golden.jsonl")

print(f"{'min_support':<12}{'refusal_acc':<13}{'spurious':<10}{'facts':<8}{'answered_wrong':<15}")
for threshold in [0.15, 0.20, 0.25, 0.28, 0.30, 0.35, 0.40]:
    pipeline = RagPipeline.from_dir(root / "data/docs", min_support=threshold)
    report = evaluate(pipeline, cases, k=5)
    m = report.metrics
    answered_wrong = sum(1 for c in report.cases if not c.answerable and not c.refused)
    print(
        f"{threshold:<12}{m['refusal_accuracy']:<13}{m['spurious_refusals']:<10}{m['fact_coverage']:<8}{answered_wrong:<15}"
    )

print()
print("best_support per case (answerable vs not):")
for threshold in [0.35]:
    pipeline = RagPipeline.from_dir(root / "data/docs", min_support=threshold)
    report = evaluate(pipeline, cases, k=5)
    ans = sorted(c.best_support for c in report.cases if c.answerable)
    gap = sorted(c.best_support for c in report.cases if not c.answerable)
    print(f"  answerable  min={min(ans):.2f} p25={ans[len(ans) // 4]:.2f} median={ans[len(ans) // 2]:.2f}")
    print(f"  unanswerable values: {[round(v, 2) for v in gap]}")
