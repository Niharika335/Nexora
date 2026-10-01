"""Multi-intent gate: a deterministic, deliberately liberal check for whether to run the planner.

A false positive costs one planner call; a false negative loses a sub-intent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List

from slrag.nlp.lemmas import REQUEST_VERBS, WH_WORDS, content_tokens, tokens

# Split points between list items / clauses.
COORD_SPLIT_RE = re.compile(r"\s*(?:,\s*(?:and|or|plus|also)\b|;|\?|\band\b|\bas well as\b|\bplus\b|\balso\b|,)\s*", re.IGNORECASE)
LIST_CUE_RE = re.compile(r"\b(?:and|also|plus|as well as)\b|,", re.IGNORECASE)
CLAUSE_OPENERS = WH_WORDS | REQUEST_VERBS | {"can", "could", "would", "is", "are", "does", "do", "i", "we", "please"}


@dataclass
class GateDecision:
    run_planner: bool
    reasons: List[str] = field(default_factory=list)


def segments(buffer: str) -> List[str]:
    return [s.strip() for s in COORD_SPLIT_RE.split(buffer) if s and s.strip()]


def starts_clause(segment: str) -> bool:
    toks = [t.lower() for t in tokens(segment)]
    return bool(toks) and toks[0] in CLAUSE_OPENERS


def request_clause_count(buffer: str) -> int:
    """Segments that open a request/question and carry at least one content token of their own."""
    return sum(1 for s in segments(buffer) if starts_clause(s) and content_tokens(s))


def coordinated_topic_count(buffer: str) -> int:
    """Number of topic phrases joined by and / , / as well as / plus (segments with content tokens)."""
    if not LIST_CUE_RE.search(buffer):
        return 0
    return sum(1 for s in segments(buffer) if content_tokens(s))


def multi_intent_gate(buffer: str) -> GateDecision:
    reasons: List[str] = []
    if coordinated_topic_count(buffer) >= 2:
        reasons.append("coordinated_noun_phrases")
    if request_clause_count(buffer) >= 2:
        reasons.append("request_clauses")
    if len(content_tokens(buffer)) >= 14 and LIST_CUE_RE.search(buffer):
        reasons.append("long_with_list_cue")
    return GateDecision(run_planner=bool(reasons), reasons=reasons)
