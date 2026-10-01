"""Generic prompts for the Phase 7 delta planner and rewriter (domain-agnostic).

The reply format is described inline by field name. The JSON schema itself is not repeated in the
prompt: json_call validates every reply against it, and the Ollama backend enforces it as `format`."""
from __future__ import annotations

import json
from typing import Any, Dict, List

DELTA_PLANNER_PROMPT = """You maintain a grounded answer made of claims, grouped by sub-question.
The user just said something new. Decide how it relates to the existing answer:
- "modifies": it changes the conditions of an existing sub-question (dates, exceptions, scope, quantities);
- "adds": it asks a new question in the same conversation;
- "unrelated": it starts a new topic.
Name the ids of the sub-questions it affects (only for "modifies"), and propose at most 2 short
retrieval queries that capture only what is new. Reply with a JSON object with the fields
"relation" ("modifies" | "adds" | "unrelated"), "affected_sub_intents" (list of sub-question ids)
and "delta_queries" (list of at most 2 strings).

New statement: {utterance}
Sub-questions: {sub_intents}
Claims (first 20 words, with cited chunk ids): {claims}"""

REWRITER_PROMPT = """You revise a grounded answer after the user added a constraint.
For every claim below choose "keep" (still correct under the constraint), "revise" (give the
corrected text and cites) or "retract" (no longer true). You may add up to 3 new claims for the
listed sub-questions. Use only the evidence below and cite only its chunk ids.
Reply with a JSON object with the fields "decisions" (one per claim: "claim_id", "action" =
"keep" | "revise" | "retract", and for "revise" the new "text" and "cites") and "new_claims"
(at most {max_new_claims}, each with "sub_intent_id", "text" and "cites").

Constraint: {constraint}
Claims: {claims}
New evidence: {evidence}"""


def first_words(text: str, n: int = 20) -> str:
    return " ".join(text.split()[:n])


def delta_planner_prompt(utterance: str, sub_intents: List[Dict[str, Any]], claims: List[Dict[str, Any]]) -> str:
    return DELTA_PLANNER_PROMPT.format(
        utterance=utterance, sub_intents=json.dumps(sub_intents, ensure_ascii=False),
        claims=json.dumps(claims, ensure_ascii=False),
    )


def rewriter_prompt(constraint: str, claims: List[Dict[str, Any]], evidence: List[Dict[str, Any]], max_new_claims: int = 3) -> str:
    return REWRITER_PROMPT.format(
        constraint=constraint, claims=json.dumps(claims, ensure_ascii=False),
        evidence=json.dumps(evidence, ensure_ascii=False), max_new_claims=max_new_claims,
    )
