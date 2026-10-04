# grounded-rag-lab

A small, dependency-free reference implementation of **citation-first RAG with
guardrails and an offline evaluation harness**.

Most RAG demos show a happy-path question and a fluent paragraph. This one shows
the parts that decide whether the thing is usable at all: every sentence carries
a source label, the system **refuses** when the corpus does not support an
answer, and the claimed behaviour is **measured** against a golden set that
includes questions the corpus *cannot* answer.

```console
$ raglab ask "What is the API rate limit on the Growth plan?"
────────────────────────────────────────────────────────────────────────
Q: What is the API rate limit on the Growth plan?
────────────────────────────────────────────────────────────────────────
Plan: Growth  |  Sustained limit: 300 requests per minute  |  Burst
allowance: 2x for 10 seconds [S1]. Rate limits are applied per workspace and
reset every 60 seconds [S1]. Plan: Scale  |  Sustained limit: 1,200 requests
per minute  |  Burst allowance: 2x for 10 seconds [S1].

support=1.00  citations=1
  [S1] rate-limits-and-errors · API rate limits and errors > Rate limits
      "Rate limits are applied per workspace and reset every 60 seconds."

$ raglab ask "Is there a mobile app for iOS and Android?"
────────────────────────────────────────────────────────────────────────
REFUSED  (reason: low_retrieval_support)
The sources do not support an answer, so the system declines instead of guessing.
```

*Both transcripts are real output (`--json` from the same run is in
`reports/eval-results.jsonl`).*

Measured on this repository's own golden set - **34 cases: 27 answerable, 7 that
must be refused** - reproducible with `make eval`:

| metric | value | what it means |
| --- | --- | --- |
| Retrieval recall@5 | **1.00** | the expected document was retrieved for every answerable question |
| MRR | **1.00** | and ranked first every time |
| Fact coverage | **0.97** | expected facts present in the answers (36 of 37) |
| Citation precision | **0.87** | citations pointing at the expected document |
| Invalid citations | **0** | hallucinated labels that survived the guardrail (target: 0) |
| Refusal accuracy | **1.00** | answered what is answerable AND refused what is not |
| Spurious refusals | **0** | answerable questions refused anyway |
| Answer support | **1.00** | share of answer sentences backed by their citation |
| Latency p50 / p95 | **≈0.8 / ≈1.3 ms** | end-to-end, offline answerer, single core |

`docs/index.html` is a generated, self-contained report: headline metrics, the
per-question traces, the retrieved passages and the guardrail log, with no CDN,
no framework and no network access.

## Quickstart

```bash
make install          # venv + editable install (uv if present, else python -m venv)
make ingest           # chunk and index data/docs
make ask Q="How long are audit logs retained?"
make eval             # golden set -> reports/*.md|jsonl + docs/index.html
make test             # 41 tests
make lint             # ruff check + format check
```

Python 3.10+ (CI covers 3.10, 3.11, 3.12). The core has **zero runtime
dependencies** and runs fully offline - no API key, no model download.

## How it works

```
data/docs/*.md ─▶ chunking ─▶ hybrid retrieval ─▶ grounded answerer ─▶ Answer
                 (markdown      BM25 ⊕ TF-IDF      (3 guardrails +      (text,
                  aware)         fused with RRF)    citations)           citations,
                                                                        trace)
                                    golden set ─▶ evaluation harness ─▶ reports
```

**Chunking** (`chunking.py`) - heading-scoped sections packed into
sentence-aligned overlapping windows, so no chunk ends mid-sentence. Markdown
table rows are folded into `Header: value` lines: a bare `| Growth | $199 |`
row cannot be retrieved meaningfully without its header. Every chunk keeps its
heading path (`Billing > Refunds > Timing`), which is prepended when indexing.

**Retrieval** (`bm25.py`, `embedding.py`, `retrieval.py`) - Okapi BM25 fused
with TF-IDF cosine similarity through Reciprocal Rank Fusion. Fusing *ranks*
sidesteps having to normalise two incomparable score scales. Exact identifiers
(error codes, plan names) come from BM25, paraphrase from the vector side.

**Grounding** (`answer.py`) - three guardrails, each covered by a test that
asserts it fires exactly when it should:

| guardrail | fires when | behaviour |
| --- | --- | --- |
| `low_support` | the best passage does not cover the question (support < 0.35) | refuse **before** calling the model |
| `no_valid_citation` | the draft cites a label that is not in the context | drop the label, retry once with a stricter prompt, then refuse |
| `unsupported_sentence` | a sentence is not backed by the source it cites | report it, or strip it with `--strict` |

`support = 0.5 × vector similarity + 0.5 × query-term coverage` is a plain,
inspectable number rather than a model's self-assessment. `Answer.trace` records
retrieval scores, which rule fired, and what was dropped - so a single answer
can be audited:

```bash
raglab ask "How long is the free trial and does it require a credit card?" --json
```

**Offline by default** (`llm.py`) - the shipped answerer is extractive: it
selects the sentences its retrieved passages actually contain, making the
benchmark deterministic and ~1 ms per question. With `OPENAI_API_KEY` and
`--remote` the same pipeline, prompt and guardrails run against any
OpenAI-compatible endpoint (`RAGLAB_LLM_MODEL`, `OPENAI_BASE_URL`).

**Evaluation** (`evaluate.py`) - one golden set, three outputs (markdown for
humans, JSONL for machines, HTML for the browser), and metrics that include the
failure modes rather than only the successes.

## Why the guardrails exist

A wrong-but-confident answer costs more than no answer when the corpus is a
contract, a pricing page or a runbook. So refusal is checked *before* generation,
citations are a hard contract, and an unsupported sentence is surfaced as a
defect instead of being smoothed over.

## Tuning the refusal threshold

`min_support` is not a guess: `scripts/tune_threshold.py` sweeps it and prints
how the two populations separate on this corpus.

```console
$ python scripts/tune_threshold.py
min_support refusal_acc  spurious  facts   answered_wrong
0.20        0.9412       0         0.973   2
0.28        0.9706       0         0.973   1
0.30        1.0          0         0.973   0
0.35        1.0          0         0.973   0
0.40        0.9706       1         0.9459  0

answerable  min=0.39 p25=0.53 median=0.59
unanswerable values: [0.0, 0.0, 0.14, 0.16, 0.17, 0.27, 0.29]
```

The default (0.35) sits inside the plateau between the highest unanswerable
score (0.29) and the lowest answerable one (0.39): below ~0.28 the system starts
answering questions the corpus cannot support, above 0.40 it starts refusing
ones it can.

## Honest limitations

* The offline answerer is **extractive**: it can only quote, and it prefers
  sentences from the passage that best supports the question. That is the right
  trade for a reproducible benchmark, not for production prose - use `--remote`
  for that. It also means one fact out of 37 in the golden set (fact coverage
  0.97) is left to a neighbouring sentence the selector did not pick.
* The third guardrail compares the answer's terms with the cited source. It
  catches invented sentences and wrong labels; it cannot detect a *subtle
  misreading* of a genuine sentence. That needs an entailment model.
* Embeddings default to TF-IDF so the demo needs no downloads and no keys. Swap
  in a real encoder by implementing the two-method `Embedder` protocol.
* The corpus is **synthetic** (a fictional product, "Ledgerline"; 5 documents,
  29 chunks). Recall is therefore 1.00 - on this index the interesting numbers
  are refusal accuracy, citation validity and answer support, not retrieval
  difficulty.
* Only English is measured. The tokenizer handles CJK as character bigrams, but
  there is no Chinese golden set yet.

## Repository layout

```
src/raglab/        14 files, ~2.1k lines: text, chunking, bm25, embedding,
                   retrieval, llm, answer, pipeline, evaluate, report, cli
data/docs/         synthetic product documentation (markdown, with tables)
data/eval/         golden.jsonl - answerable cases with expected facts + gaps
scripts/           tune_threshold.py, the sweep behind the default threshold
tests/             41 tests (453 lines): text, chunking, retrieval, guardrails,
                   metrics, CLI
docs/index.html    generated evaluation report (publishable as static HTML)
reports/           eval-report.md + eval-results.jsonl (machine readable)
.github/workflows/ ci: lint, tests on 3 Python versions, quality gates
```

CI fails if the guarantees regress: `invalid_citations` and
`spurious_refusals` must stay 0, and recall, fact coverage, refusal accuracy and
answer support must stay above 0.95.

## License

MIT © 2026 Shuo Zhao. The corpus was written for this repository; no third-party
data and no client material are included.
