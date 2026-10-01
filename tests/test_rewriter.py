"""Phase 7 rewriter: rewrite scope, enum-constrained cites, decision parsing and fallbacks."""

import asyncio
from dataclasses import dataclass
import json

import pytest

from slrag.answer.rewriter import Rewriter, RewriteScopeError, allowed_cites
from slrag.config import DEFAULT_CONFIG
from slrag.contracts.events import Claim, ClaimStatus
from slrag.pipeline.verifier import FailClosedVerifier


@dataclass
class Chunk:
    chunk_id: str
    text: str


def claim(cid, sub, text, cites):
    return Claim(claim_id=cid, sub_intent_id=sub, text=text, doc_ids=cites, verification_status=ClaimStatus.VERIFIED)


C1 = claim("c1", "sub_1", "Refunds are issued within 14 days.", ["P§1"])
C2 = claim("c2", "sub_1", "Refunds go to the original card.", ["P§1"])
C3 = claim("c3", "sub_2", "Invoices are emailed monthly.", ["I§1"])
DELTA = {"sub_1": [Chunk("P§2", "International refunds are issued within 30 days."), Chunk("P§3", "Fees apply abroad.")]}


def fixed(output):
    async def fn(prompt, schema, context):
        return output if isinstance(output, str) else json.dumps(output)
    return fn


def rewrite(fn, claims=(C1, C2), affected=("sub_1",), held=None):
    return asyncio.run(Rewriter(fn).rewrite("The purchase was international.", list(affected), list(claims), DELTA, held or {"sub_1": ["P§1"]}))


def test_only_affected_claims_can_reach_the_rewriter():
    with pytest.raises(RewriteScopeError):
        rewrite(fixed({"decisions": [], "new_claims": []}), claims=(C1, C3))
    seen = {}

    async def spy(prompt, schema, context):
        seen["ids"] = [c["claim_id"] for c in context["claims"]]
        seen["prompt"] = prompt
        return json.dumps({"decisions": [], "new_claims": []})

    res = rewrite(spy)
    assert res.input_claim_ids == ["c1", "c2"] == seen["ids"]
    assert C3.text not in seen["prompt"]


def test_allowed_cites_are_affected_cites_plus_delta_plus_held_evidence():
    assert allowed_cites([C1, C2], DELTA, {"sub_1": ["P§0"]}) == ["P§0", "P§1", "P§2", "P§3"]
    res = rewrite(fixed({"decisions": [], "new_claims": []}), held={"sub_1": ["P§0"], "sub_2": ["I§1"]})
    assert res.allowed_cites == ["P§0", "P§1", "P§2", "P§3"]  # held evidence of unaffected sub-intents is not allowed


def test_parses_keep_revise_retract_and_new_claims():
    res = rewrite(fixed({
        "decisions": [
            {"claim_id": "c1", "action": "revise", "text": "International refunds are issued within 30 days.", "cites": ["P§2"]},
            {"claim_id": "c2", "action": "retract"},
        ],
        "new_claims": [{"sub_intent_id": "sub_1", "text": "Fees apply abroad.", "cites": ["P§3"]}],
    }))
    assert not res.fallback
    assert [(d.claim_id, d.action, d.cites) for d in res.decisions] == [("c1", "revise", ["P§2"]), ("c2", "retract", None)]
    assert [(n.sub_intent_id, n.cites) for n in res.new_claims] == [("sub_1", ["P§3"])]


def test_unmentioned_claims_are_kept():
    res = rewrite(fixed({"decisions": [{"claim_id": "c2", "action": "keep"}], "new_claims": []}))
    assert [(d.claim_id, d.action) for d in res.decisions] == [("c1", "keep"), ("c2", "keep")]


@pytest.mark.parametrize("output, reason", [
    # Enum test: a cite outside the allowed set is rejected at schema level.
    ({"decisions": [{"claim_id": "c1", "action": "revise", "text": "x", "cites": ["Z§9"]}], "new_claims": []}, "schema"),
    ({"decisions": [], "new_claims": [{"sub_intent_id": "sub_1", "text": "x", "cites": ["I§1"]}]}, "schema"),
    # An unaffected claim id or sub-intent is not in the enum.
    ({"decisions": [{"claim_id": "c3", "action": "keep"}], "new_claims": []}, "schema"),
    ({"decisions": [], "new_claims": [{"sub_intent_id": "sub_2", "text": "x", "cites": ["P§2"]}]}, "schema"),
    ({"decisions": [], "new_claims": [{"sub_intent_id": "sub_1", "text": "x", "cites": []}]}, "schema"),
    ({"decisions": [{"claim_id": "c1", "action": "rewrite"}], "new_claims": []}, "schema"),
    ({"decisions": [{"claim_id": "c1", "action": "keep"}, {"claim_id": "c1", "action": "retract"}], "new_claims": []}, "duplicate_decision"),
    ({"decisions": [{"claim_id": "c1", "action": "revise"}], "new_claims": []}, "revise_without_text"),
    ("{broken", "invalid_json"),
])
def test_invalid_output_keeps_every_claim(output, reason):
    res = rewrite(fixed(output))
    assert res.fallback and res.fallback_reason == reason
    assert [d.action for d in res.decisions] == ["keep", "keep"] and res.new_claims == []


def test_verifier_check_a_rejects_cites_outside_the_allowed_set():
    check = FailClosedVerifier(DEFAULT_CONFIG.verifier).check_hallucinated_id(["Z§9"], {"P§1", "P§2"})
    assert not check.passed


def test_evidence_is_capped_per_affected_sub_intent():
    many = {"sub_1": [Chunk(f"P§{i}", f"text {i}") for i in range(10)]}
    res = asyncio.run(Rewriter(fixed({"decisions": [], "new_claims": []}), evidence_per_sub_intent=4)
                      .rewrite("c", ["sub_1"], [C1], many, {}))
    assert res.evidence_sent == {"sub_1": ["P§0", "P§1", "P§2", "P§3"]}
