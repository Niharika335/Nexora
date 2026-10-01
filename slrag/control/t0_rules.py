"""T0: presentation-turn rules (evaluated only once the session ledger has an active claim)."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable, List, Optional, Set

from slrag.nlp.lemmas import CorpusVocab, anchors, content_lemmas, lemma, tokens

PRESENTATION_VERBS = {
    "repeat", "shorten", "summarize", "summarise", "rephrase", "reword", "translate", "bullet", "bullets",
    "simplify", "reformat", "condense", "restate", "tl;dr", "tldr", "shorter", "briefly",
}
ANAPHORA_WORDS = {"that", "this", "it", "those", "them", "above", "previous", "earlier"}
ANAPHORA_PHRASES = ("last answer", "your answer", "what you said")
# Formatting vocabulary is an instruction parameter, never a content anchor ("two bullets").
FORMAT_NOUNS = {
    "bullet", "bullets", "point", "points", "line", "lines", "sentence", "sentences", "word", "words",
    "paragraph", "paragraphs", "item", "items", "step", "steps", "answer", "version", "list",
}
PRESENTATION_VOCAB = PRESENTATION_VERBS | ANAPHORA_WORDS | FORMAT_NOUNS | {"last", "your", "please", "again"}


@dataclass
class T0Result:
    score: float
    verb: int
    anaph: int
    noanchor: int
    new_anchors: List[str] = field(default_factory=list)
    route: str = "continue"  # suppress | continue | ambiguous


def ledger_vocabulary(texts: Iterable[str]) -> Set[str]:
    """Lower-cased tokens and lemmas of sub-intent texts and active claim texts."""
    vocab: Set[str] = set()
    for text in texts:
        vocab |= {t.lower() for t in tokens(text)}
        vocab |= content_lemmas(text)
    return vocab


def _format_quantities(buffer: str) -> Set[str]:
    """Numbers that directly precede a formatting noun ("3 bullets", "two sentences")."""
    toks = [t.lower() for t in tokens(buffer)]
    return {toks[i] for i in range(len(toks) - 1) if toks[i + 1] in FORMAT_NOUNS}


def t0_score(
    buffer: str,
    ledger_vocab: Set[str],
    vocab: Optional[CorpusVocab] = None,
    suppress_at: float = 0.70,
    continue_below: float = 0.40,
) -> T0Result:
    low = buffer.lower()
    toks = [t.lower() for t in tokens(buffer)]

    verb = int(any(t in PRESENTATION_VERBS for t in toks) or "tl;dr" in low)
    anaph = int(any(t in ANAPHORA_WORDS for t in toks) or any(re.search(rf"\b{p}\b", low) for p in ANAPHORA_PHRASES))

    skip = _format_quantities(buffer)
    new_anchors = [
        a for a in anchors(buffer, vocab)
        if a not in skip and a not in PRESENTATION_VOCAB and a not in ledger_vocab and lemma(a) not in ledger_vocab
    ]
    noanchor = int(not new_anchors)

    score = 0.5 * verb + 0.3 * anaph + 0.2 * noanchor
    if score >= suppress_at and noanchor:
        route = "suppress"
    elif score < continue_below:
        route = "continue"
    else:
        route = "ambiguous"
    return T0Result(score=round(score, 3), verb=verb, anaph=anaph, noanchor=noanchor, new_anchors=new_anchors, route=route)
