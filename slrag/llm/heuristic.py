"""Deterministic stand-in backends for the Phase 7 json_calls.

The project has no real LLM yet (LLMServiceWrapper extracts sentences from chunks), so these
play the model's role for replay and tests: they read the structured `context` that
accompanies the prompt and return JSON text. Their output goes through exactly the same
schema validation, enum checks, verification and fallbacks a real model's output would, so a
real backend can be swapped in by passing a different JsonFn. Their decisions are lexical
heuristics, not language understanding; relation accuracy is measured against gold labels.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Set, Tuple

from slrag.nlp.lemmas import REQUEST_VERBS, WH_WORDS, content_lemmas, split_sentences, tokens

# Words that frame a constraint rather than carry its content.
CONSTRAINT_CUES = {
    "assume", "assuming", "suppose", "consider", "also", "account", "case", "given", "instead",
    "now", "actually", "note", "remember", "keep", "mind", "make", "sure", "treat", "answer",
}
EXCLUSION_RE = re.compile(r"\b(?:ignore|ignoring|exclude|excluding|except|without|no longer|drop)\s+((?:\w+[\s-]?){1,3})", re.IGNORECASE)


def _core_lemmas(text: str) -> Set[str]:
    return content_lemmas(text) - CONSTRAINT_CUES


def is_question(utterance: str) -> bool:
    toks = [t.lower() for t in tokens(utterance)]
    return utterance.strip().endswith("?") or bool(toks and (toks[0] in WH_WORDS or toks[0] in REQUEST_VERBS))


def core_query(utterance: str, limit: int = 160) -> str:
    """The utterance without framing words ("Assume", "Also consider", ...)."""
    words = [t for t in tokens(utterance) if t.lower() not in CONSTRAINT_CUES]
    return " ".join(words)[:limit].strip() or utterance[:limit]


async def heuristic_delta_llm(prompt: str, schema: Dict[str, Any], context: Dict[str, Any]) -> str:
    utterance: str = context["utterance"]
    u = _core_lemmas(utterance)
    scores: List[Tuple[int, str]] = []
    for sub in context.get("sub_intents", []):
        claims = [c for c in context.get("claims", []) if c["sub_intent_id"] == sub["id"]]
        # Claim openings plus the words of the documents they cite ("nx-feature-copilot§..." -> copilot).
        claim_text = " ".join([c["first_20_words"] for c in claims]
                              + [re.sub(r"[^A-Za-z0-9]+", " ", cite) for c in claims for cite in c.get("cites") or []])
        scores.append((len(u & (content_lemmas(sub["text"]) | content_lemmas(claim_text))), sub["id"]))
    top = max((s for s, _ in scores), default=0)

    if is_question(utterance):
        relation, affected = ("adds" if top >= 2 else "unrelated"), []
    else:
        relation = "modifies"
        affected = [sid for s, sid in scores if top > 0 and s == top][:2]
    return json.dumps({"relation": relation, "affected_sub_intents": affected, "delta_queries": [core_query(utterance)]})


async def heuristic_rewrite_llm(prompt: str, schema: Dict[str, Any], context: Dict[str, Any]) -> str:
    constraint: str = context["constraint"]
    claims: List[Dict[str, Any]] = context["claims"]
    claim_lemmas = {c["claim_id"]: content_lemmas(c["text"]) for c in claims}
    known = set().union(*claim_lemmas.values()) if claim_lemmas else set()
    new_info = _core_lemmas(constraint) - known
    need = min(2, len(new_info)) or 1

    excluded: Set[str] = set()
    for m in EXCLUSION_RE.finditer(constraint):
        excluded |= content_lemmas(m.group(1))

    existing_texts = {c["text"] for c in claims}
    candidates: List[Tuple[int, int, Dict[str, Any]]] = []
    for order, ev in enumerate(context.get("evidence", [])):
        for sent in split_sentences(ev["text"]):
            score = len(content_lemmas(sent) & new_info)
            if score >= need and sent not in existing_texts:
                candidates.append((-score, order, {"text": sent, "chunk_id": ev["chunk_id"], "sub_intent_id": ev["sub_intent_id"]}))
    candidates.sort(key=lambda x: (x[0], x[1]))

    decisions: Dict[str, Dict[str, Any]] = {}
    for c in claims:
        if excluded and claim_lemmas[c["claim_id"]] & excluded:
            decisions[c["claim_id"]] = {"claim_id": c["claim_id"], "action": "retract"}

    new_claims: List[Dict[str, Any]] = []
    for _, _, cand in candidates:
        target = next(
            (c for c in claims
             if c["claim_id"] not in decisions and c["sub_intent_id"] == cand["sub_intent_id"]
             and len(claim_lemmas[c["claim_id"]] & content_lemmas(cand["text"])) >= 3),
            None,
        )
        if target is not None:
            decisions[target["claim_id"]] = {
                "claim_id": target["claim_id"], "action": "revise", "text": cand["text"], "cites": [cand["chunk_id"]],
            }
        elif len(new_claims) < context.get("max_new_claims", 3) and cand["text"] not in {n["text"] for n in new_claims}:
            new_claims.append({"sub_intent_id": cand["sub_intent_id"], "text": cand["text"], "cites": [cand["chunk_id"]]})

    for c in claims:
        decisions.setdefault(c["claim_id"], {"claim_id": c["claim_id"], "action": "keep"})
    return json.dumps({"decisions": [decisions[c["claim_id"]] for c in claims], "new_claims": new_claims})


class HeuristicLLM:
    """The default backend (llm.backend: heuristic): deterministic stand-ins for every LLM role.

    Drafting is extractive (LLMServiceWrapper); the delta planner and rewriter use the lexical
    json functions above. No network access, so tests and CI never need a model server."""

    mode = "heuristic"
    delta_fn = staticmethod(heuristic_delta_llm)
    rewrite_fn = staticmethod(heuristic_rewrite_llm)

