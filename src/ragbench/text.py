"""Lightweight, dependency-free text utilities used by retrievers and heuristic judges."""

from __future__ import annotations

import re
import string

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers herself him himself his how i if in into is it
    its itself just me more most my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours yourself yourselves
    also may must per s t""".split()
)

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[.'\-][A-Za-z0-9]+)*")
_DIGIT_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3})")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n+")
_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")

ABSTENTION_PATTERNS = re.compile(
    r"(i don'?t know|i do not know|not (?:enough|sufficient) information|"
    r"(?:context|documents?|sources?) (?:does|do) not (?:contain|provide|mention|say|include)|"
    r"cannot be (?:answered|determined)|can'?t be (?:answered|determined)|"
    r"no information (?:about|on|regarding)|unable to (?:answer|determine|find))",
    re.IGNORECASE,
)


def stem(token: str) -> str:
    """Very light plural stemming; enough to match 'days'/'day', 'policies'/'policy'."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens. Thousands separators are removed (``1,150`` -> ``1150``)."""
    text = _DIGIT_COMMA_RE.sub("", text)
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def content_tokens(text: str) -> list[str]:
    """Tokens with stopwords removed and light stemming applied."""
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS]


def is_number(token: str) -> bool:
    return bool(_NUMBER_RE.match(token))


def split_sentences(text: str) -> list[str]:
    parts = (p.strip(" \t-*#") for p in _SENT_SPLIT_RE.split(text))
    return [p for p in parts if p and any(c.isalnum() for c in p)]


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token) for providers without usage data."""
    return max(1, round(len(text) / 4)) if text else 0


def is_abstention(answer: str) -> bool:
    return bool(ABSTENTION_PATTERNS.search(answer or ""))


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation/articles/whitespace."""
    text = _DIGIT_COMMA_RE.sub("", text.lower())
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())
