"""Static HTML report / interactive demo player.

The page is a single self-contained file: no CDN, no build step, no server. It
replays *real* pipeline traces (the answers and retrieval signals produced by
``raglab eval``) so a reviewer can see what the system actually did, including
the runs where it refused to answer.

Data is embedded as JSON and rendered through ``textContent`` only, so the page
cannot be used to inject markup.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from .evaluate import EvalReport
from .types import Answer

_PLACEHOLDER = "__RAGLAB_DATA__"

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>grounded-rag-lab &middot; evaluation report</title>
<style>
  :root{
    --bg:#0b0f14; --panel:#121821; --panel-2:#0f141c; --line:#1f2733; --ink:#e6edf3;
    --muted:#8b97a6; --accent:#4ea1ff; --ok:#3fb950; --warn:#d29922; --bad:#f85149;
    --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  header{padding:28px 32px 18px;border-bottom:1px solid var(--line);background:linear-gradient(180deg,#0e141d,#0b0f14)}
  h1{margin:0 0 6px;font-size:22px;letter-spacing:.2px}
  header p{margin:0;color:var(--muted);font-size:13.5px}
  .wrap{max-width:1180px;margin:0 auto;padding:22px 24px 60px}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:10px;margin:18px 0 26px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
  .card .k{color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.06em}
  .card .v{font-size:21px;font-family:var(--mono);margin-top:4px}
  .card .d{color:var(--muted);font-size:11.5px;margin-top:2px}
  .layout{display:grid;grid-template-columns:340px 1fr;gap:20px;align-items:start}
  @media (max-width:900px){.layout{grid-template-columns:1fr}}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
  .panel h2{margin:0;padding:12px 14px;font-size:12.5px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);border-bottom:1px solid var(--line);background:var(--panel-2)}
  ul.qs{list-style:none;margin:0;padding:6px}
  ul.qs li{padding:9px 10px;border-radius:8px;cursor:pointer;font-size:13.5px;color:#c9d4e0}
  ul.qs li:hover{background:#182231}
  ul.qs li.active{background:#1b2a3d;color:#fff}
  ul.qs li .tag{display:inline-block;font-size:10.5px;font-family:var(--mono);padding:1px 6px;border-radius:999px;margin-right:7px;border:1px solid var(--line);color:var(--muted)}
  ul.qs li .tag.ans{color:var(--ok);border-color:#1d3a24}
  ul.qs li .tag.ref{color:var(--warn);border-color:#3b2f12}
  .detail{padding:18px 20px}
  .qhead{font-size:16.5px;font-weight:600;margin:0 0 4px}
  .qmeta{color:var(--muted);font-size:12.5px;font-family:var(--mono);margin-bottom:14px}
  .answer{border-left:3px solid var(--accent);background:#0f1723;padding:12px 14px;border-radius:0 8px 8px 0;white-space:pre-wrap}
  .answer.refused{border-left-color:var(--warn);background:#171307}
  .cite{display:inline-block;font-family:var(--mono);font-size:12px;background:#1b2a3d;color:#9ecbff;padding:1px 6px;border-radius:5px;margin-left:4px}
  h3{font-size:12.5px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:22px 0 8px}
  .src{border:1px solid var(--line);border-radius:9px;margin-bottom:8px;background:var(--panel-2)}
  .src .head{display:flex;justify-content:space-between;gap:10px;padding:8px 12px;border-bottom:1px solid var(--line);font-family:var(--mono);font-size:12px;color:#a9b6c6}
  .src .body{padding:10px 12px;font-size:13.5px;color:#c8d3df;white-space:pre-wrap}
  .src.cited{border-color:#24405f}
  .bars{display:flex;gap:8px;flex-wrap:wrap;font-family:var(--mono);font-size:11.5px;color:var(--muted)}
  .bar{display:flex;align-items:center;gap:6px}
  .bar i{display:block;height:6px;width:64px;background:#22303f;border-radius:4px;position:relative;overflow:hidden}
  .bar i b{position:absolute;inset:0 auto 0 0;background:var(--accent);border-radius:4px}
  table.trace{width:100%;border-collapse:collapse;font-size:12.5px}
  table.trace td,table.trace th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
  table.trace th{color:var(--muted);font-weight:500;text-transform:uppercase;font-size:11px;letter-spacing:.05em}
  code{font-family:var(--mono);font-size:12px;background:#182231;padding:1px 5px;border-radius:4px}
  .pill{font-family:var(--mono);font-size:11.5px;padding:1px 7px;border-radius:999px;border:1px solid var(--line)}
  .pill.pass{color:var(--ok);border-color:#1d3a24}
  .pill.refuse{color:var(--warn);border-color:#3b2f12}
  .pill.strip{color:#d2a8ff;border-color:#33234a}
  footer{color:var(--muted);font-size:12px;margin-top:26px;border-top:1px solid var(--line);padding-top:14px}
  a{color:var(--accent)}
</style>
</head>
<body>
<header>
  <h1>grounded-rag-lab &middot; retrieval, guardrails and refusal quality</h1>
  <p id="subtitle"></p>
</header>
<div class="wrap">
  <div class="cards" id="cards"></div>
  <div class="layout">
    <div class="panel">
      <h2>Questions in this run</h2>
      <ul class="qs" id="qs"></ul>
    </div>
    <div class="panel"><div class="detail" id="detail"></div></div>
  </div>
  <footer>
    Replayed traces from <code>raglab eval</code>. Every number on this page was produced by the
    code in this repository; nothing is hand-written. Documents are synthetic.
  </footer>
</div>
<script>
const DATA = __RAGLAB_DATA__;
const el = (id) => document.getElementById(id);

function bar(label, value, max) {
  const wrap = document.createElement('div');
  wrap.className = 'bar';
  const name = document.createElement('span');
  name.textContent = label;
  const track = document.createElement('i');
  const fill = document.createElement('b');
  const pct = Math.max(0, Math.min(1, max ? value / max : 0));
  fill.style.width = (pct * 100).toFixed(1) + '%';
  track.appendChild(fill);
  const num = document.createElement('span');
  num.textContent = value.toFixed(3);
  wrap.append(name, track, num);
  return wrap;
}

function renderCards() {
  const cards = el('cards');
  DATA.metrics.forEach((m) => {
    const c = document.createElement('div');
    c.className = 'card';
    const k = document.createElement('div'); k.className = 'k'; k.textContent = m.label;
    const v = document.createElement('div'); v.className = 'v'; v.textContent = m.value;
    const d = document.createElement('div'); d.className = 'd'; d.textContent = m.note || '';
    c.append(k, v, d);
    cards.appendChild(c);
  });
}

function renderList() {
  const list = el('qs');
  DATA.cases.forEach((c, i) => {
    const li = document.createElement('li');
    if (i === DATA.selected) li.className = 'active';
    const tag = document.createElement('span');
    tag.className = 'tag ' + (c.refused ? 'ref' : 'ans');
    tag.textContent = c.kind;
    const text = document.createElement('span');
    text.textContent = c.question;
    li.append(tag, text);
    li.onclick = () => { DATA.selected = i; renderList(); renderDetail(); };
    list.appendChild(li);
  });
}

function renderDetail() {
  const c = DATA.cases[DATA.selected];
  const d = el('detail');
  d.textContent = '';

  const q = document.createElement('p'); q.className = 'qhead'; q.textContent = c.question;
  const meta = document.createElement('div'); meta.className = 'qmeta';
  meta.textContent = `${c.expected} | ${c.latency_ms} ms | support ${c.support.toFixed(2)} | ` +
    (c.refused ? `refused: ${c.reason}` : `citations ${c.citations.map((x) => x.label).join(' ') || 'none'}`);
  d.append(q, meta);

  const answer = document.createElement('div');
  answer.className = 'answer' + (c.refused ? ' refused' : '');
  answer.textContent = c.refused
    ? `Refused to answer (${c.reason}). The corpus does not support an answer, and guessing is the failure mode this project exists to avoid.`
    : c.answer;
  d.appendChild(answer);

  if (c.unsupported.length) {
    const h = document.createElement('h3'); h.textContent = 'Sentences the guardrail stripped'; d.appendChild(h);
    const p = document.createElement('div'); p.className = 'src';
    const b = document.createElement('div'); b.className = 'body';
    b.textContent = c.unsupported.join('\\n');
    p.appendChild(b); d.appendChild(p);
  }

  const h1 = document.createElement('h3'); h1.textContent = `Retrieved passages (top ${c.retrieved.length})`; d.appendChild(h1);
  c.retrieved.forEach((r) => {
    const box = document.createElement('div');
    box.className = 'src' + (c.citations.some((x) => x.label === r.label) ? ' cited' : '');
    const head = document.createElement('div'); head.className = 'head';
    const left = document.createElement('span');
    left.textContent = `${r.label}  ${r.doc}  ·  ${r.heading}`;
    const right = document.createElement('span');
    right.className = 'bars';
    right.append(bar('sim', r.similarity, 1), bar('cov', r.coverage, 1), bar('support', r.support, 1));
    head.append(left, right);
    const body = document.createElement('div'); body.className = 'body'; body.textContent = r.text;
    box.append(head, body);
    d.appendChild(box);
  });

  if (c.citations.length) {
    const h2 = document.createElement('h3'); h2.textContent = 'Citations returned to the caller'; d.appendChild(h2);
    const table = document.createElement('table'); table.className = 'trace';
    const thead = document.createElement('tr');
    ['label', 'document', 'section', 'supporting sentence'].forEach((t) => {
      const th = document.createElement('th'); th.textContent = t; thead.appendChild(th);
    });
    table.appendChild(thead);
    c.citations.forEach((cite) => {
      const tr = document.createElement('tr');
      [cite.label, cite.doc, cite.heading, cite.quote].forEach((v) => {
        const td = document.createElement('td'); td.textContent = v; tr.appendChild(td);
      });
      table.appendChild(tr);
    });
    d.appendChild(table);
  }

  const h3 = document.createElement('h3'); h3.textContent = 'Guardrail trace'; d.appendChild(h3);
  const t2 = document.createElement('table'); t2.className = 'trace';
  const head = document.createElement('tr');
  ['rule', 'action', 'detail'].forEach((t) => { const th = document.createElement('th'); th.textContent = t; head.appendChild(th); });
  t2.appendChild(head);
  c.guardrails.forEach((g) => {
    const tr = document.createElement('tr');
    const c1 = document.createElement('td'); c1.textContent = g.rule;
    const c2 = document.createElement('td');
    const pill = document.createElement('span'); pill.className = 'pill ' + g.action; pill.textContent = g.action;
    c2.appendChild(pill);
    const c3 = document.createElement('td'); c3.textContent = JSON.stringify(g.detail);
    tr.append(c1, c2, c3);
    t2.appendChild(tr);
  });
  d.appendChild(t2);
}

el('subtitle').textContent = DATA.subtitle;
renderCards();
renderList();
renderDetail();
</script>
</body>
</html>
"""


def _case_payload(answer: Answer, expected: str, kind: str) -> dict:
    deterministic = answer.trace.get("guardrails", [])
    return {
        "question": answer.question,
        "kind": kind,
        "expected": expected,
        "refused": answer.refused,
        "reason": answer.reason,
        "answer": answer.text,
        "support": answer.support,
        "latency_ms": round(answer.trace.get("timings_ms", {}).get("total", 0.0), 1),
        "unsupported": answer.unsupported_sentences,
        "citations": [
            {"label": c.label, "doc": c.doc_id, "heading": c.heading, "quote": c.quote}
            for c in answer.citations
        ],
        "retrieved": [
            {
                "label": r.label,
                "doc": r.chunk.doc_id,
                "heading": r.chunk.heading,
                "similarity": round(r.similarity, 4),
                "coverage": round(r.query_coverage, 4),
                "support": r.support,
                "text": r.chunk.text,
            }
            for r in answer.retrieved
        ],
        "guardrails": deterministic,
    }


def render_html(
    answers: Sequence[tuple[Answer, str, str]],
    report: EvalReport,
    title: str = "grounded-rag-lab",
) -> str:
    """Build the self-contained HTML page.

    ``answers`` is a sequence of (Answer, expected-docs-label, kind) tuples.
    """
    from .evaluate import METRIC_LABELS

    metrics = []
    for key, label, note in METRIC_LABELS:
        value = report.metrics.get(key)
        if value is None:
            continue
        metrics.append({"label": label, "value": value, "note": note})
    corpus = report.meta.get("corpus", {})
    payload = {
        "subtitle": (
            f"{corpus.get('documents', '?')} synthetic documents · {corpus.get('chunks', '?')} chunks · "
            f"embedder {corpus.get('embedder', '?')} · answerer {corpus.get('llm', '?')} · "
            f"top-k {report.meta.get('k', '?')} · {len(answers)} replayed questions"
        ),
        "metrics": metrics,
        "selected": 0,
        "cases": [_case_payload(a, expected, kind) for a, expected, kind in answers],
    }
    return _TEMPLATE.replace(_PLACEHOLDER, json.dumps(payload, ensure_ascii=False))


def write_html(
    path,
    answers: Sequence[tuple[Answer, str, str]],
    report: EvalReport,
) -> object:
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(answers, report), encoding="utf-8")
    return target
