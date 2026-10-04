"""Answer generators.

Three implementations behind one interface:

* ``ExtractiveLLM``  - offline, deterministic, no API key. Picks the sentences
  with the highest overlap with the question and cites them. This is the
  default so that anyone (or any CI job) can run the whole pipeline.
* ``OpenAICompatLLM`` - any OpenAI-compatible chat endpoint, with a system
  prompt that makes the citation contract explicit.
* ``ScriptedLLM``    - returns canned text, used by the tests to exercise the
  guardrails with a *misbehaving* model (hallucinated citations, no citations,
  invented facts).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .text import content_tokens, coverage, split_sentences

_TABLE_ROW_RE = re.compile(r"^\|")
_TABLE_RULE_RE = re.compile(r"^\|[\s:|-]+\|$")
_HAS_DIGIT_RE = re.compile(r"\d")

# Question cues that mean "the answer is a number somewhere in the source".
_QUANTITY_CUES = (
    "how long",
    "how much",
    "how many",
    "how often",
    "how frequently",
    "what is the price",
    "cost",
    "price",
    "rate",
    "limit",
    "quota",
    "target",
    "window",
    "retention",
    "expire",
    "day",
    "hour",
    "minute",
    "month",
    "year",
    "percent",
)


def _wants_quantity(question: str) -> bool:
    lowered = question.lower()
    return any(cue in lowered for cue in _QUANTITY_CUES)


def _prettify(sentence: str) -> str:
    """Turn a markdown table row into a readable line; leave prose untouched."""
    stripped = sentence.strip()
    if not _TABLE_ROW_RE.match(stripped):
        return stripped
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    cells = [cell for cell in cells if cell]
    return " · ".join(cells) if cells else stripped


NOT_FOUND = "NOT_FOUND"


@dataclass(frozen=True)
class Source:
    """A retrieved passage handed to the model, with the label it must cite."""

    label: str  # "[S1]"
    doc_id: str
    heading: str
    text: str


class LLMClient(Protocol):
    name: str

    def generate(self, question: str, sources: Sequence[Source], strict: bool = False) -> str: ...


def _finish(sentence: str) -> str:
    return sentence if sentence.endswith((".", "!", "?", "。", "！", "？", ":")) else sentence + "."


def cite(sentence: str, label: str) -> str:
    """Attach a citation to a sentence *before* its final punctuation.

    Placement matters downstream: the support guardrail splits the answer into
    sentences, and a marker written after the full stop would be attributed to
    the following sentence instead of the one it supports.
    """
    text = sentence.strip()
    if text.endswith((".", "!", "?", "。", "！", "？", ":", ";")):
        return f"{text[:-1]} {label}{text[-1]}"
    return f"{text} {label}."


class ExtractiveLLM:
    """Deterministic offline answerer - the default, needs no credentials."""

    name = "extractive-offline"

    def __init__(
        self,
        max_sentences: int = 3,
        min_sentences: int = 3,
        min_sentence_score: float = 0.15,
        keep_ratio: float = 0.5,
    ) -> None:
        self.max_sentences = max_sentences
        self.min_sentences = min_sentences
        self.min_sentence_score = min_sentence_score
        self.keep_ratio = keep_ratio

    def generate(self, question: str, sources: Sequence[Source], strict: bool = False) -> str:
        query = content_tokens(question)
        if not query or not sources:
            return NOT_FOUND

        # Term weights from the context itself: a question term that shows up in
        # many sentences (e.g. "plan", "limit") carries less signal than a rare
        # one (e.g. "growth"), so matches are scored by specificity.
        sentences: list[tuple[int, Source, list[str]]] = [
            (order, source, split_sentences(source.text)) for order, source in enumerate(sources)
        ]
        document_frequency: dict[str, int] = {}
        for _, _, source_sentences in sentences:
            for sentence in source_sentences:
                for term in set(content_tokens(sentence)):
                    document_frequency[term] = document_frequency.get(term, 0) + 1
        weights = {term: 1.0 / (1.0 + document_frequency.get(term, 0)) for term in set(query)}
        # Only terms that actually occur in this context may dilute the score:
        # a question term that appears nowhere says nothing about which sentence
        # is the best answer.
        present = [term for term in set(query) if document_frequency.get(term, 0) > 0]
        total_weight = sum(weights[term] for term in present) or sum(weights.values()) or 1.0
        wants_quantity = _wants_quantity(question)

        candidates: list[tuple[float, int, str, str]] = []
        for order, source, source_sentences in sentences:
            heading_bonus = 0.05 * coverage(query, source.heading)
            for sentence in source_sentences:
                if _TABLE_RULE_RE.match(sentence.strip()):
                    continue  # "| --- | --- |" separator rows carry no content
                tokens = set(content_tokens(sentence))
                if len(tokens) < 3:
                    continue
                matched = tokens & set(query)
                specificity = sum(weights[term] for term in matched) / total_weight
                quantity_bonus = 0.1 if (wants_quantity and _HAS_DIGIT_RE.search(sentence)) else 0.0
                candidates.append(
                    (specificity + heading_bonus + quantity_bonus, order, source.label, _prettify(sentence))
                )
        if not candidates:
            return NOT_FOUND
        candidates.sort(key=lambda item: (-item[0], item[1], item[3]))
        if candidates[0][0] < self.min_sentence_score:
            return NOT_FOUND
        # Only keep sentences that are close to the best match, so a weak match
        # does not drag unrelated rows into the answer.
        cutoff = max(self.min_sentence_score, self.keep_ratio * candidates[0][0])
        passing = [c for c in candidates if c[0] >= cutoff]
        # Prefer the passage that best supports the question. Without this, a
        # generic phrase ("growth plan limits") can pull a sentence from an
        # unrelated section into an otherwise focused answer. Sentences from
        # other passages only top up a thin answer.
        lead = candidates[0][1]
        selected = [c for c in passing if c[1] == lead][: self.max_sentences]
        if len(selected) < self.min_sentences:
            for candidate in passing:
                if candidate[1] == lead:
                    continue
                selected.append(candidate)
                if len(selected) >= self.min_sentences:
                    break
        selected.sort(key=lambda item: (item[1], -item[0]))
        return " ".join(cite(sentence, label) for _, _, label, sentence in selected)


class ScriptedLLM:
    """Test double: replays a fixed list of outputs (last one repeats)."""

    name = "scripted"

    def __init__(self, outputs: Sequence[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def generate(self, question: str, sources: Sequence[Source], strict: bool = False) -> str:
        self.calls.append({"question": question, "strict": strict, "labels": [s.label for s in sources]})
        index = min(len(self.calls) - 1, len(self.outputs) - 1)
        return self.outputs[index] if self.outputs else NOT_FOUND


class OpenAICompatLLM:
    """Any OpenAI-compatible /chat/completions endpoint (OpenAI, DeepSeek, Qwen, vLLM...)."""

    SYSTEM_PROMPT = (
        "You answer strictly from the numbered sources provided by the user.\n"
        "Rules:\n"
        "1. Every claim must end with the label of the source that supports it, e.g. [S2].\n"
        "2. Never use outside knowledge and never invent a label that is not in the sources.\n"
        "3. If the sources do not contain the answer, reply with exactly NOT_FOUND and nothing else.\n"
        "4. Be concise: at most three sentences."
    )
    STRICT_REMINDER = (
        "Your previous answer was rejected: it had no valid [S#] citation. "
        "Answer again, citing at least one existing source label after every sentence. "
        "If that is impossible, reply exactly NOT_FOUND."
    )

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        self.name = f"openai-compat:{model or os.environ.get('RAGLAB_LLM_MODEL', 'gpt-4o-mini')}"
        self.model = model or os.environ.get("RAGLAB_LLM_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip(
            "/"
        )
        self.temperature = temperature
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("OpenAICompatLLM needs OPENAI_API_KEY (or pass api_key=...)")

    def _render(self, question: str, sources: Sequence[Source]) -> str:
        blocks = "\n\n".join(f"{s.label} (doc: {s.doc_id} | section: {s.heading})\n{s.text}" for s in sources)
        return f"Sources:\n\n{blocks}\n\nQuestion: {question}"

    def generate(self, question: str, sources: Sequence[Source], strict: bool = False) -> str:
        user = self._render(question, sources)
        if strict:
            user = f"{user}\n\n{self.STRICT_REMINDER}"
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode())
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise RuntimeError(f"chat completion failed: {exc}") from exc
        return (body["choices"][0]["message"]["content"] or "").strip()


def default_llm(prefer_remote: bool = False) -> LLMClient:
    """Remote model when requested and keyed, otherwise the offline answerer."""
    if prefer_remote and os.environ.get("OPENAI_API_KEY"):
        return OpenAICompatLLM()
    return ExtractiveLLM()
