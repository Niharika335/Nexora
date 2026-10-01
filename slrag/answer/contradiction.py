"""Contradiction guard between revised/new claims and preserved claims (Phase 7, HIGH VALUE).

For a candidate claim and each preserved claim sharing >= 2 content lemmas, extract
(noun lemma, number) pairs: spaCy `nummod` arcs when spaCy and en_core_web_sm are installed,
otherwise a regex fallback (a number followed by its next content word). If the same noun is
quantified with a number the preserved claim does not state, the candidate is flagged; the
caller downgrades it to an uncertainty item and keeps the preserved claim.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from slrag.nlp.lemmas import STOPWORDS, content_lemmas, is_number, lemma, normalize_number, numbers, tokens

Pair = Tuple[str, str]  # (noun lemma, normalized number)


@lru_cache(maxsize=1)
def _spacy_nlp() -> Optional[Any]:
    try:
        import spacy  # type: ignore

        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def backend() -> str:
    return "spacy" if _spacy_nlp() is not None else "regex"


def _pairs_spacy(nlp: Any, text: str) -> Set[Pair]:
    pairs: Set[Pair] = set()
    for tok in nlp(text):
        if tok.dep_ == "nummod" and tok.head.pos_ in ("NOUN", "PROPN"):
            pairs.add((lemma(tok.head.lemma_), normalize_number(tok.text)))
    return pairs


def _pairs_regex(text: str) -> Set[Pair]:
    toks = tokens(text)
    pairs: Set[Pair] = set()
    for i, tok in enumerate(toks):
        if not is_number(tok):
            continue
        for nxt in toks[i + 1:i + 3]:  # "30 attendees", "64 kilobytes", "five node cluster"
            if is_number(nxt):
                break
            if nxt.lower() not in STOPWORDS:
                pairs.add((lemma(nxt), normalize_number(tok)))
                break
    return pairs


def noun_number_pairs(text: str, use_spacy: bool = True) -> Set[Pair]:
    nlp = _spacy_nlp() if use_spacy else None
    return _pairs_spacy(nlp, text) if nlp is not None else _pairs_regex(text)


def find_contradiction(candidate_text: str, preserved: Iterable[Any], use_spacy: bool = True) -> Optional[Dict[str, Any]]:
    """First preserved claim the candidate contradicts, or None.

    `preserved` items need `.text` and `.claim_id` (Claim models)."""
    cand_words = content_lemmas(candidate_text) - numbers(candidate_text)
    cand_pairs = noun_number_pairs(candidate_text, use_spacy)
    if not cand_pairs:
        return None
    for prev in preserved:
        prev_words = content_lemmas(prev.text) - numbers(prev.text)
        if len(cand_words & prev_words) < 2:
            continue
        prev_pairs = noun_number_pairs(prev.text, use_spacy)
        for noun in {n for n, _ in cand_pairs}:
            cand_nums = {v for n, v in cand_pairs if n == noun}
            prev_nums = {v for n, v in prev_pairs if n == noun}
            if prev_nums and not cand_nums <= prev_nums:
                return {
                    "preserved_claim_id": prev.claim_id, "noun": noun,
                    "candidate_numbers": sorted(cand_nums), "preserved_numbers": sorted(prev_nums),
                    "backend": "spacy" if use_spacy and _spacy_nlp() is not None else "regex",
                }
    return None


def guard(candidates: List[Tuple[str, str]], preserved: List[Any], use_spacy: bool = True) -> Dict[str, Dict[str, Any]]:
    """Check (key, text) candidates; returns {key: contradiction} for the flagged ones."""
    flags: Dict[str, Dict[str, Any]] = {}
    for key, text in candidates:
        hit = find_contradiction(text, preserved, use_spacy)
        if hit is not None:
            flags[key] = hit
    return flags
