"""Fallback splitter used when the LLM planner is unavailable, times out, or returns invalid JSON.

Uses spaCy's dependency parse when spaCy and `en_core_web_sm` are installed, otherwise a
lexical heuristic that splits (a) at coordinators that open a new question/request clause
and (b) at noun-phrase lists governed by a request verb ("I need the X and the Y").
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any, List, Optional

from slrag.nlp.lemmas import DETERMINERS, PREPOSITIONS, REQUEST_VERBS, WH_WORDS, content_tokens, entities, tokens
from slrag.plan.multi_intent_gate import COORD_SPLIT_RE, starts_clause

PRONOUN_RE = re.compile(r"\b(it|they|them)\b", re.IGNORECASE)
NP_VERB_RE = re.compile(
    r"^(?P<prefix>.*?\b(?:" + "|".join(sorted(REQUEST_VERBS, key=len, reverse=True)) + r")\b(?:\s+(?:me|us))?)(?P<rest>\s.+)$",
    re.IGNORECASE,
)
NP_LIST_SPLIT_RE = re.compile(r"\s*(?:,\s*and\b|,\s*|\band\b|\bas well as\b|\bplus\b)\s*", re.IGNORECASE)
POSSESSIVE_STARTS = DETERMINERS | {"his", "her"}


@dataclass
class SplitResult:
    parts: List[str]
    preamble: str = ""  # text before the governing request verb of an NP list (carries shared context)
    method: str = "heuristic"


def _clean(text: str) -> str:
    return text.strip().strip(",;").strip()


def _split_clauses(buffer: str) -> List[str]:
    """Split at coordinators/punctuation only where the right side opens a new clause."""
    pieces: List[tuple] = []
    last = 0
    for m in COORD_SPLIT_RE.finditer(buffer):
        pieces.append((buffer[last:m.start()], m.group(0)))
        last = m.end()
    pieces.append((buffer[last:], ""))

    parts: List[str] = []
    current, pending_sep = "", ""
    for text, sep in pieces:
        if not text.strip():
            pending_sep += sep
            continue
        if current and starts_clause(text) and content_tokens(text) and content_tokens(current):
            parts.append(_clean(current + ("?" if "?" in pending_sep else "")))
            current = text
        else:
            current = current + pending_sep + text if current else text
        pending_sep = sep
    if current.strip():
        parts.append(_clean(current + ("?" if "?" in pending_sep else "")))
    return parts


def _split_np_list(clause: str) -> Optional[SplitResult]:
    """'I need the cancellation policy and the badge printing options' -> one part per noun phrase."""
    m = NP_VERB_RE.match(clause.rstrip("?.! "))
    if not m:
        return None
    rest = m.group("rest")
    if any(t.lower() in WH_WORDS for t in tokens(rest)):
        return None
    items = [i.strip() for i in NP_LIST_SPLIT_RE.split(rest) if i and i.strip()]
    if len(items) < 2 or not all(content_tokens(i) for i in items):
        return None
    comma_list = rest.count(",") >= 1 and len(items) >= 3
    first_words = [tokens(i)[0].lower() if tokens(i) else "" for i in items]
    determiners_on_all = all(w in POSSESSIVE_STARTS for w in first_words)
    same_preposition = first_words[0] in PREPOSITIONS and len(set(first_words)) == 1  # "about X and about Y"
    if not (comma_list or determiners_on_all or same_preposition):
        return None
    prefix = m.group("prefix").strip()
    verb_match = re.search(r"\b(" + "|".join(REQUEST_VERBS) + r")\b", prefix, re.IGNORECASE)
    preamble = prefix[: verb_match.start()] if verb_match else ""
    return SplitResult(parts=[f"{prefix} {item}" for item in items], preamble=preamble)


def topic_of(text: str) -> str:
    ents = entities(text)
    if ents:
        return ents[0]
    content = content_tokens(text)
    return " ".join(content[:4])


def resolve_pronouns(parts: List[str]) -> List[str]:
    """Replace a standalone it/they/them in later parts with the first part's topic."""
    if len(parts) < 2:
        return parts
    topic = topic_of(parts[0])
    if not topic:
        return parts
    return [parts[0]] + [PRONOUN_RE.sub(topic, p, count=1) for p in parts[1:]]


@lru_cache(maxsize=1)
def _spacy_nlp() -> Any:
    import spacy  # optional dependency

    return spacy.load("en_core_web_sm")


def _spacy_split(buffer: str) -> Optional[SplitResult]:
    """Dependency-based split: clause conjuncts, else the conj chain of the last request verb's object."""
    try:
        doc = _spacy_nlp()(buffer)
    except Exception:
        return None

    # Clause-level coordination: a coordinating conjunction whose right conjunct is a verb.
    verb_conjs = [t for t in doc if t.dep_ == "conj" and t.pos_ in ("VERB", "AUX") and t.head.pos_ in ("VERB", "AUX")]
    if verb_conjs:
        cut_points = sorted(min(w.i for w in c.subtree) for c in verb_conjs)
        spans, start = [], 0
        for cut in cut_points:
            left = doc[start:cut]
            while len(left) and left[-1].dep_ in ("cc", "punct"):
                left = left[:-1]
            spans.append(left.text)
            start = cut
        spans.append(doc[start:].text)
        parts = [_clean(s) for s in spans if content_tokens(s)]
        if len(parts) >= 2:
            return SplitResult(parts=parts, method="spacy")

    # Noun-phrase coordination under the last request verb.
    req = [t for t in doc if t.lemma_.lower() in REQUEST_VERBS and t.pos_ == "VERB"]
    if req:
        verb = req[-1]
        objs = [c for c in verb.children if c.dep_ in ("dobj", "attr", "pobj", "obj")]
        if objs:
            chain = [objs[0]]
            while True:
                nxt = [c for c in chain[-1].children if c.dep_ == "conj"]
                if not nxt:
                    break
                chain.append(nxt[0])
            if len(chain) >= 2:
                prefix = doc[: verb.i + 1].text
                items = [doc[noun.left_edge.i: noun.i + 1].text for noun in chain]
                return SplitResult(parts=[f"{prefix} {i}" for i in items], preamble=doc[: verb.i].text, method="spacy")
    return None


def is_request(text: str) -> bool:
    """A question or request clause (vs. declarative context such as 'I am planning a trip to X')."""
    toks = [t.lower() for t in tokens(text)]
    if not toks:
        return False
    return (
        toks[0] in WH_WORDS | REQUEST_VERBS | {"can", "could", "would", "is", "are", "does", "do"}
        or any(t in REQUEST_VERBS for t in toks[:3])
        or text.rstrip().endswith("?")
    )


def _separate_preamble(parts: List[str]) -> tuple:
    """Leading non-request clauses become shared context when request clauses follow them."""
    lead = 0
    while lead < len(parts) and not is_request(parts[lead]):
        lead += 1
    if 0 < lead < len(parts):
        return " ".join(parts[:lead]), parts[lead:]
    return "", parts


def split_fallback(buffer: str, use_spacy: bool = True) -> SplitResult:
    """Split a compound utterance into self-contained parts; returns [buffer] if no coordination is found."""
    if use_spacy:
        try:
            spacy_result = _spacy_split(buffer)
        except Exception:
            spacy_result = None
        if spacy_result is not None:
            return SplitResult(resolve_pronouns(spacy_result.parts), spacy_result.preamble, spacy_result.method)

    preamble, clauses = _separate_preamble(_split_clauses(buffer))
    parts: List[str] = []
    for clause in clauses:
        np_split = _split_np_list(clause)
        if np_split is not None:
            parts.extend(np_split.parts)
            preamble = " ".join(p for p in (preamble, np_split.preamble.strip()) if p)
        else:
            parts.append(clause)
    if not parts:
        parts = [buffer.strip()]
    return SplitResult(parts=resolve_pronouns(parts), preamble=preamble)
