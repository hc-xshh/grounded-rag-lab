"""Tokenisation and sentence splitting shared by every component.

Kept deliberately simple and dependency-free; the ASCII path handles normal
English technical text and the CJK path adds character bigrams so that
Chinese documents are retrievable too.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'’._-]*")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|\n{2,}")
_TERMINAL = (".", "!", "?", ":", ";", "。", "！", "？", "：", "；")
_STRUCTURAL = ("|", "-", "*", "•", "#", ">", "+")
# marker used when a markdown table row is expanded into "Header: value" pairs
TABLE_ROW_SEPARATOR = "  |  "


def _ends_sentence(line: str) -> bool:
    """True when the next line must not be appended to this one."""
    return line.endswith(_TERMINAL) or TABLE_ROW_SEPARATOR in line


# Terms too common to carry any retrieval signal.
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "should",
    "so",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "us",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
}


def _stem(token: str) -> str:
    """Very light English suffix stripping.

    Enough to match "processed" with "process" and "requests" with "request",
    which matters because questions and documents rarely agree on inflection.
    Deliberately conservative: the stem must stay at least four characters, and
    "es" is only dropped when English actually uses it that way (boxes, watches).
    """
    if len(token) >= 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) >= 5 and token.endswith("ed"):
        return token[:-2]
    if len(token) >= 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) >= 5 and token.endswith("es") and token[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return token[:-2]
    if len(token) >= 4 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens (lightly stemmed) plus CJK character bigrams.

    Stopwords are removed *before* stemming so that inflected forms such as
    "does" cannot survive as artificial content terms.
    """
    lowered = text.lower()
    tokens: list[str] = []
    for raw in _WORD_RE.findall(lowered):
        token = raw.strip("._-")  # sentence-final punctuation sticks to the word
        if not token or token in STOPWORDS:
            continue
        stem = _stem(token)
        if stem in STOPWORDS:
            continue
        tokens.append(stem)
    for run in _CJK_RE.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def content_tokens(text: str) -> list[str]:
    """Tokens carrying retrieval signal (stopwords and single characters dropped)."""
    return [t for t in tokenize(text) if len(t) > 1]


def split_sentences(text: str) -> list[str]:
    """Split into sentences, undoing hard line wrapping and keeping list items intact.

    Documents are frequently wrapped at a fixed column width, which would cut
    sentences in half, so a continuation line is appended to the previous one.
    Two things must *not* be swallowed that way: a new structural element
    (bullet, table row, heading, quote) always starts its own unit, and a
    sentence that already ended stays closed.
    """
    units: list[tuple[str, str]] = []  # (text, kind) where kind is "flow" or "block"
    for block in _SENTENCE_RE.split(text):
        block = block.strip(" \t")
        if not block:
            continue
        for raw in block.splitlines():
            line = raw.strip()
            if not line:
                continue
            is_bullet = line.startswith(("-", "*", "•")) and len(line) < 240
            if is_bullet:
                units.append((line.lstrip("-*• ").strip(), "block"))
            elif line.startswith(_STRUCTURAL) or TABLE_ROW_SEPARATOR in line:
                units.append((line, "block"))
            else:
                units.append((line, "flow"))

    merged: list[str] = []
    absorbable: list[bool] = []
    for part, kind in units:
        if kind == "block" and absorbable:
            absorbable[-1] = False  # a bullet starts its own unit, never joins the previous one
        if merged and absorbable[-1]:
            merged[-1] = f"{merged[-1]} {part}"
            absorbable[-1] = not _ends_sentence(part)
        else:
            merged.append(part)
            absorbable.append(not _ends_sentence(part))
    return [part for part in merged if len(part) > 1]


def coverage(query_tokens: list[str], text: str) -> float:
    """Share of the query's distinct content terms that appear in `text`."""
    if not query_tokens:
        return 0.0
    target = set(tokenize(text))
    hits = sum(1 for t in set(query_tokens) if t in target)
    return round(hits / len(set(query_tokens)), 4)


def normalise_fact(fact: str) -> str:
    """Loose normalisation used when checking whether an expected fact is present."""
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", fact.lower()).strip()
