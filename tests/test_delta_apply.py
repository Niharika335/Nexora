"""Phase 7 ledger delta: keep / revise / retract / add, versioning, preservation, contradiction guard."""

import pytest

from slrag.answer.contradiction import find_contradiction, noun_number_pairs
from slrag.contracts.events import Claim, ClaimStatus, compute_text_hash
from slrag.state.delta import Decision, NewClaim, PreservationError, apply_delta
from slrag.state.ledger import ClaimLedger


def ledger_v1():
    """Answer v1: 4 claims in 2 sub-intents."""
    ledger = ClaimLedger("s")
    for sub, text, cite in [
        ("sub_1", "Refunds are issued within 14 days.", "P§1"),
        ("sub_1", "Refunds go to the original card.", "P§1"),
        ("sub_2", "Invoices are emailed monthly.", "I§1"),
        ("sub_2", "Invoices list the tax rate.", "I§2"),
    ]:
        ledger.add_or_update_claim(sub, text, doc_ids=[cite], turn_id="t1")
    ledger.bump_answer_version()
    return ledger


def test_keep_and_revise_on_sub_intent_1_leave_sub_intent_2_and_the_kept_claim_byte_identical():
    ledger = ledger_v1()
    before = {c.claim_id: c.text_hash for c in ledger.get_verified_claims()}
    assert ledger.answer_version == 1 and list(before) == ["c1", "c2", "c3", "c4"]

    res = apply_delta(ledger, ["sub_1"], [
        Decision("c1", "keep"),
        Decision("c2", "revise", "International refunds go to the original card within 30 days.", ["P§2"]),
    ], [], turn_id="t2", rewrite_input_claim_ids=["c1", "c2"], strict=True)

    after = {c.claim_id: c for c in ledger.get_verified_claims()}
    assert after["c3"].text_hash == before["c3"] and after["c4"].text_hash == before["c4"]  # sub-intent 2
    assert after["c1"].text_hash == before["c1"]  # kept
    assert after["c2"].text_hash != before["c2"]  # revised
    assert after["c2"].text_hash == compute_text_hash(after["c2"].text, ["P§2"])
    assert (res.from_version, res.to_version, ledger.answer_version) == (1, 2, 2)
    assert res.unchanged_hashes_ok and res.preserved_checked == res.preserved_identical == 3
    assert (res.kept, res.revised, res.unaffected) == (["c1"], ["c2"], ["c3", "c4"])
    assert after["c2"].status == "revised" and after["c2"].last_modified_version == 2
    assert after["c2"].history[-1] == {"version": 2, "action": "revise",
                                       "text": "International refunds go to the original card within 30 days.", "cites": ["P§2"]}
    assert after["c1"].status == "active"
    assert res.ops == [
        {"op": "keep", "claim_id": "c1"},
        {"op": "revise", "claim_id": "c2", "text": "International refunds go to the original card within 30 days.",
         "cites": ["P§2"], "prev_text_hash": before["c2"]},
        {"op": "keep", "claim_id": "c3"},
        {"op": "keep", "claim_id": "c4"},
    ]
    payload = res.transition_payload()
    assert {k: payload[k] for k in ("from", "to", "kept", "revised", "retracted", "added", "unchanged_hashes_ok", "rewrite_input_claim_ids")} == {
        "from": 1, "to": 2, "kept": ["c1"], "revised": ["c2"], "retracted": [], "added": [],
        "unchanged_hashes_ok": True, "rewrite_input_claim_ids": ["c1", "c2"],
    }


def test_retract_and_add():
    ledger = ledger_v1()
    res = apply_delta(ledger, ["sub_1"], [Decision("c1", "retract")],
                      [NewClaim("sub_1", "International refunds take 30 days.", ["P§2"])], turn_id="t2", strict=True)
    live = ledger.get_verified_claims()
    assert [c.claim_id for c in live] == ["c2", "c5", "c3", "c4"]  # the new claim stays in sub-intent order
    assert res.retracted == ["c1"] and res.added == ["c5"]
    assert ledger.retracted_claims[0].claim_id == "c1" and ledger.retracted_claims[0].status == "retracted"
    new = next(c for c in live if c.claim_id == "c5")
    assert new.created_version == 2 and new.verification_status == ClaimStatus.VERIFIED and new.doc_ids == ["P§2"]
    assert ledger.answer_version == 2  # exactly one bump for a retract + add
    assert {"op": "retract", "claim_id": "c1"} in res.ops
    assert {"op": "add", "claim_id": "c5", "text": "International refunds take 30 days.", "cites": ["P§2"]} in res.ops


@pytest.mark.parametrize("decisions, news", [
    ([Decision("c3", "keep")], []),  # claim of an unaffected sub-intent
    ([Decision("c9", "keep")], []),  # unknown claim
    ([Decision("c1", "keep"), Decision("c1", "retract")], []),  # duplicate
    ([Decision("c1", "revise", "")], []),  # revise without text
    ([], [NewClaim("sub_2", "x", ["I§1"])]),  # new claim outside the affected sub-intents
    ([], [NewClaim("sub_1", "x", [])]),  # new claim without cites
])
def test_invalid_deltas_are_rejected_before_anything_changes(decisions, news):
    ledger = ledger_v1()
    snapshot = ledger.create_snapshot_event("t1").model_dump(exclude={"event_id", "timestamp"})
    with pytest.raises(ValueError):
        apply_delta(ledger, ["sub_1"], decisions, news, turn_id="t2")
    assert ledger.create_snapshot_event("t1").model_dump(exclude={"event_id", "timestamp"}) == snapshot


def test_preservation_assertion_raises_in_strict_mode_and_logs_otherwise(monkeypatch, caplog):
    def tampering_replace(self, claims, retracted, turn_id):
        # Simulated bug: an unaffected claim is modified while applying the delta.
        self._verified_claims = [c.model_copy(update={"text": c.text + "!"}) if c.claim_id == "c3" else c for c in claims]

    monkeypatch.setattr(ClaimLedger, "replace_live_claims", tampering_replace)
    with pytest.raises(PreservationError):
        apply_delta(ledger_v1(), ["sub_1"], [Decision("c1", "keep")], [], turn_id="t2", strict=True)
    res = apply_delta(ledger_v1(), ["sub_1"], [Decision("c1", "keep")], [], turn_id="t2", strict=False)
    assert not res.unchanged_hashes_ok and res.preserved_identical == res.preserved_checked - 1
    assert "preservation assertion failed" in caplog.text


def test_checkpoint_restore_roundtrip():
    ledger = ledger_v1()
    cp = ledger.checkpoint()
    apply_delta(ledger, ["sub_1"], [Decision("c1", "retract")], [], turn_id="t2")
    ledger.restore(cp)
    assert [c.claim_id for c in ledger.get_verified_claims()] == ["c1", "c2", "c3", "c4"] and ledger.answer_version == 1


def test_contradiction_guard_fixture():
    preserved = Claim(claim_id="c1", text="The venue holds up to 30 attendees per workshop session.")
    flag = find_contradiction("The venue holds up to 50 attendees per workshop session.", [preserved], use_spacy=False)
    assert flag is not None and flag["preserved_claim_id"] == "c1" and flag["noun"] == "attendee"
    assert (flag["candidate_numbers"], flag["preserved_numbers"]) == (["50"], ["30"])
    # Same number, or too little shared content (< 2 lemmas), is not a contradiction.
    assert find_contradiction("The venue holds up to 30 attendees per workshop session.", [preserved], use_spacy=False) is None
    assert find_contradiction("Parking allows 50 attendees.", [preserved], use_spacy=False) is None
    assert noun_number_pairs("Raft uses five nodes and tolerates two failures.", use_spacy=False) == {("node", "5"), ("failure", "2")}
