"""Phase 3 fixes: ledger answer_version / sub_intents / text_hash, and the consistency verifier."""

from slrag.config import DEFAULT_CONFIG
from slrag.contracts.events import Claim, ClaimStatus, SubIntent
from slrag.pipeline.ledger import ClaimLedger
from slrag.pipeline.verifier import FailClosedVerifier

CITED = (
    "A Raft cluster consists of several nodes, typically five, allowing the system to tolerate two failures. "
    "The leader does not accept writes during an election."
)


def verified(text, cid="c1", cites=("DOC002§raft",)):
    return Claim(claim_id=cid, text=text, doc_ids=list(cites), verification_status=ClaimStatus.VERIFIED)


def test_ledger_answer_version_sub_intents_uncertainty_and_text_hash():
    ledger = ClaimLedger("s1")
    assert ledger.answer_version == 0
    ledger.add_sub_intent(SubIntent(id="t1:q1", text="How many failures does Raft tolerate?", status="answerable"))
    ledger.add_verified_claims([verified("Raft tolerates two failures.")], turn_id="t1")
    ledger.add_uncertainty("t1:q2", "insufficient_evidence", {"dense_top1": 0.4, "coverage": 0.2})
    assert ledger.answer_version == 1

    snap = ledger.create_snapshot_event("t1")
    assert snap.answer_version == 1
    assert [s.id for s in snap.sub_intents] == ["t1:q1"]
    assert snap.uncertainty[0]["reason"] == "insufficient_evidence"
    dumped = snap.model_dump()
    assert len(dumped["verified_claims"][0]["text_hash"]) == 64

    # A turn that adds nothing (e.g. a presentation turn) leaves answer_version unchanged.
    ledger.add_verified_claims([], turn_id="t2")
    assert ledger.answer_version == 1


def test_non_destructive_update_keeps_history():
    ledger = ClaimLedger()
    first = ledger.add_or_update_claim("q1", "Raft uses five nodes.", doc_ids=["D§1"], turn_id="t1")
    second = ledger.add_or_update_claim("q1", "Raft typically uses five nodes.", doc_ids=["D§1"], is_update=True, turn_id="t2")
    active = ledger.get_verified_claims()
    assert [c.claim_id for c in active] == [second] and first != second
    assert active[0].history[-1] == {"version": 1, "action": "revise", "from": first, "text": "Raft typically uses five nodes."}
    assert ledger._superseded_claims[0].verification_status == ClaimStatus.SUPERSEDED


def test_consistency_passes_supported_claim():
    v = FailClosedVerifier(DEFAULT_CONFIG.verifier)
    assert v.check_consistency("A Raft cluster of five nodes tolerates two failures.", [], CITED).passed


def test_consistency_rejects_negation_flip_against_cited_sentence():
    v = FailClosedVerifier(DEFAULT_CONFIG.verifier)
    res = v.check_consistency("The leader accepts writes during an election.", [], CITED)
    assert not res.passed and "Negation" in res.details


def test_consistency_rejects_number_absent_from_evidence():
    v = FailClosedVerifier(DEFAULT_CONFIG.verifier)
    res = v.check_consistency("A Raft cluster of seven nodes tolerates three failures.", [], CITED)
    assert not res.passed and "Numbers" in res.details


def test_consistency_rejects_contradiction_with_ledger():
    v = FailClosedVerifier(DEFAULT_CONFIG.verifier)
    prior = [verified("The leader does not accept client writes during an election.", cid="c7")]
    cited = "The leader accepts client writes during an election once it is elected."
    res = v.check_consistency("The leader accepts client writes during an election.", prior, cited)
    assert not res.passed and "c7" in res.details


def test_consistency_ignores_unrelated_negative_ledger_claims():
    """Regression: the old precedence bug rejected any claim that was a substring of a negative prior claim."""
    v = FailClosedVerifier(DEFAULT_CONFIG.verifier)
    prior = [verified("Raft does not require a majority quorum for reads.", cid="c3")]
    assert v.check_consistency("Raft", prior, "Raft is a consensus algorithm.").passed


def test_verify_claim_fails_closed_on_consistency():
    v = FailClosedVerifier(DEFAULT_CONFIG.verifier)
    chunk = type("C", (), {"text": CITED})()
    claim = Claim(text="A Raft cluster consists of several nodes, typically seven, allowing the system to tolerate two failures.",
                  doc_ids=["DOC002§raft"])
    passed, checks, reason = v.verify_claim(claim, {"DOC002§raft": chunk}, {"DOC002§raft"}, [])
    assert passed is False and checks["consistency"].passed is False and "consistency" in reason
