"""Query builder: Q_t from the live utterance buffer."""
from __future__ import annotations

from dataclasses import dataclass

from slrag.nlp.lemmas import STOPWORDS, has_dangling_tail, strip_dangling_tail, tokens


@dataclass(frozen=True)
class BuiltQuery:
    query: str  # content tokens in utterance order
    content_count: int
    dangling: bool


def build_query(buffer: str) -> BuiltQuery:
    """Keep content tokens (and entity casing) in order, dropping a trailing dangling tail."""
    dangling = has_dangling_tail(buffer)
    toks = strip_dangling_tail(tokens(buffer))
    content = [t for t in toks if t.lower() not in STOPWORDS]
    return BuiltQuery(query=" ".join(content), content_count=len(content), dangling=dangling)
