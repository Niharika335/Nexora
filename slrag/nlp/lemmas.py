"""Lightweight lexical NLP: content tokens, anchors, slots, dangling tails, negation.

spaCy is not a hard dependency, so these are lexicon/regex heuristics standing in for
POS tags and NER. Entities are capitalised or alphanumeric tokens (not sentence-initial
unless acronym-like), quantities come from number patterns, and domain nouns are corpus
vocabulary terms whose IDF is at least the median IDF.
"""

from __future__ import annotations

import re
import statistics
from typing import Dict, Iterable, List, Mapping, Optional, Set

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[\-\.'][A-Za-z0-9]+)*%?")

DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "some", "any", "each", "every", "my", "your", "our", "their", "its"}
PREPOSITIONS = {
    "in", "on", "at", "for", "to", "of", "with", "by", "from", "about", "into", "onto", "over",
    "under", "between", "across", "through", "during", "via", "per", "within", "without", "around", "near",
}
COORDINATORS = {"and", "or", "but", "nor", "plus", "also"}
AUXILIARIES = {
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "have", "has", "had",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
}
PRONOUNS = {"i", "me", "you", "he", "she", "it", "we", "they", "them", "us", "him", "her", "what's", "it's"}
WH_WORDS = {"what", "how", "why", "which", "who", "whom", "whose", "when", "where"}
FILLERS = {"um", "uh", "erm", "hmm", "so", "well", "like", "okay", "ok", "yeah", "please", "just", "actually", "basically"}
REQUEST_VERBS = {
    "need", "want", "know", "tell", "check", "find", "explain", "list", "describe", "summarize",
    "summarise", "compare", "show", "give", "define", "outline", "detail",
}
STOPWORDS = (
    DETERMINERS | PREPOSITIONS | COORDINATORS | AUXILIARIES | PRONOUNS | WH_WORDS | FILLERS
    | {"not", "no", "than", "then", "there", "here", "as", "if", "so", "very", "too", "more", "most",
       "such", "other", "own", "same", "only", "both", "all", "again", "let", "me", "please"}
)
DANGLING_TAIL = DETERMINERS | PREPOSITIONS | COORDINATORS | AUXILIARIES | {"…", "..."}

NEGATION_CUES = {
    "not", "no", "never", "cannot", "without", "none", "neither", "nor", "unable", "prohibited", "ineligible",
}
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "hundred": 100, "thousand": 1000,
}
MONTHS = {
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december",
}
NUMBER_RE = re.compile(r"^\$?\d+(?:[.,]\d+)*%?$")


def tokens(text: str) -> List[str]:
    """Surface tokens with original casing."""
    return TOKEN_RE.findall(text)


def lemma(token: str) -> str:
    """Crude lemmatiser: lowercase and strip regular plural endings."""
    t = token.lower()
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("ses"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss") and not t[-2].isdigit():
        return t[:-1]
    return t


def content_tokens(text: str) -> List[str]:
    """Lower-cased tokens that are not function words or fillers."""
    return [t.lower() for t in tokens(text) if t.lower() not in STOPWORDS]


def content_lemmas(text: str) -> Set[str]:
    return {lemma(t) for t in content_tokens(text)}


def has_dangling_tail(text: str) -> bool:
    """True if the utterance currently ends mid-phrase (preposition, determiner, conjunction, aux, ellipsis)."""
    stripped = text.rstrip()
    if stripped.endswith("…") or stripped.endswith("..."):
        return True
    toks = tokens(stripped)
    if not toks:
        return False
    if stripped[-1] in ".?!":
        return False
    return toks[-1].lower() in DANGLING_TAIL


def strip_dangling_tail(toks: List[str]) -> List[str]:
    out = list(toks)
    while out and out[-1].lower() in DANGLING_TAIL:
        out.pop()
    return out


def is_number(token: str) -> bool:
    return bool(NUMBER_RE.match(token)) or token.lower() in NUMBER_WORDS


def normalize_number(token: str) -> str:
    t = token.lower().replace(",", "").replace("$", "").rstrip("%")
    if t in NUMBER_WORDS:
        return str(NUMBER_WORDS[t])
    return t


def numbers(text: str) -> Set[str]:
    return {normalize_number(t) for t in tokens(text) if is_number(t)}


def _sentence_initial_positions(text: str) -> Set[int]:
    positions = set()
    for m in re.finditer(r"(?:^|[.!?]\s+)([A-Za-z0-9])", text):
        positions.add(m.start(1))
    return positions


def _acronym_like(token: str) -> bool:
    has_digit = any(c.isdigit() for c in token)
    has_alpha = any(c.isalpha() for c in token)
    upper_count = sum(1 for c in token if c.isupper())
    return (has_digit and has_alpha) or upper_count >= 2


def entities(text: str) -> List[str]:
    """Entity-like spans: acronyms / alphanumerics anywhere, capitalised words when not sentence-initial."""
    initial = _sentence_initial_positions(text)
    found: List[str] = []
    for m in TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok.lower() in STOPWORDS or is_number(tok):
            continue
        if _acronym_like(tok) or (tok[0].isupper() and m.start() not in initial):
            if tok not in found:
                found.append(tok)
    return found


class CorpusVocab:
    """Corpus vocabulary backed by the BM25 IDF table (domain-noun detection)."""

    def __init__(self, idf_table: Mapping[str, float]):
        self._idf = dict(idf_table)
        self.median_idf = statistics.median(self._idf.values()) if self._idf else 0.0

    def has(self, token: str) -> bool:
        t = token.lower()
        return t in self._idf or lemma(t) in self._idf

    def idf(self, token: str) -> float:
        t = token.lower()
        return self._idf.get(t, self._idf.get(lemma(t), 0.0))

    def is_domain_noun(self, token: str) -> bool:
        t = token.lower()
        return t not in STOPWORDS and not is_number(t) and self.has(t) and self.idf(t) >= self.median_idf


def extract_slots(text: str) -> Dict[str, str]:
    """Typed slot values used for cache conflict checks, e.g. {"CARDINAL": "30", "ENTITY": "growth"}.

    Multiple values of one type are joined in sorted order so two queries conflict whenever
    they carry a different set of values for the same type.
    """
    typed: Dict[str, Set[str]] = {}
    for tok in tokens(text):
        low = tok.lower()
        if tok.endswith("%"):
            typed.setdefault("PERCENT", set()).add(normalize_number(tok))
        elif tok.startswith("$"):
            typed.setdefault("MONEY", set()).add(normalize_number(tok))
        elif re.fullmatch(r"(1[89]|20)\d\d", tok):
            typed.setdefault("DATE", set()).add(tok)
        elif low in MONTHS and low != "may":
            typed.setdefault("DATE", set()).add(low)
        elif is_number(tok):
            typed.setdefault("CARDINAL", set()).add(normalize_number(tok))
    ents = [e.lower() for e in entities(text)]
    if ents:
        typed["ENTITY"] = set(ents)
    return {k: "|".join(sorted(v)) for k, v in typed.items()}


def slot_conflict(a: Mapping[str, str], b: Mapping[str, str]) -> Optional[str]:
    """Return the first slot type present in both with different values, else None."""
    for key in sorted(set(a) & set(b)):
        if a[key] != b[key]:
            return f"{key}:{a[key]}!={b[key]}"
    return None


def anchors(text: str, vocab: Optional[CorpusVocab] = None) -> List[str]:
    """Anchors = entities + quantities + (if a vocab is given) domain nouns, lower-cased, deduplicated."""
    found: List[str] = []

    def add(values: Iterable[str]) -> None:
        for v in values:
            v = v.lower()
            if v not in found:
                found.append(v)

    add(entities(text))
    add(t for t in tokens(text) if is_number(t))
    if vocab is not None:
        add(t for t in content_tokens(text) if vocab.is_domain_noun(t))
    return found


def has_head_verb(text: str) -> bool:
    low = [t.lower() for t in tokens(text)]
    return any(t in REQUEST_VERBS or t in WH_WORDS for t in low)


def has_negation(text: str) -> bool:
    low = text.lower()
    if "n't" in low or re.search(r"\bnon-", low):
        return True
    return any(t.lower() in NEGATION_CUES for t in tokens(text))


def split_sentences(text: str) -> List[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return parts or [text.strip()]
