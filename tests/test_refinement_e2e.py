"""Phase 7 end-to-end: late-detail refinement through the replay TurnEngine (ours) vs restart (B1)."""

import asyncio
import json
from pathlib import Path

import pytest

from slrag.eval.clock import ReplayClock
from slrag.eval.gates import GateEvaluator
from slrag.eval.runner import InMemoryBus, ReplayRunner
from slrag.nlp.lemmas import split_sentences
from slrag.replay.baselines import build_index, build_turn_engine, mode_config

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ROOT / "eval" / "scenarios"
TURN1 = "What does the AI agenda builder do? and which Zapier triggers are available?"
TURN2 = "Assume we are on Starter with only 50 Copilot requests per month."
DELTA_GOLD = "nx-feature-copilot§copilot-limits"


@pytest.fixture(scope="module")
def index():
    return build_index(mode_config("ours").app)


def session(index, mode="ours", **kw):
    bus, clock = InMemoryBus(), ReplayClock()
    engine = build_turn_engine(mode, index, bus, clock=clock, overrides={"refinement.strict_preservation": True}, **kw)
    return engine, bus, clock


def run(engine, clock, query, turn_id, **kw):
    clock.advance(2.0)
    return asyncio.run(engine.execute_turn(query, turn_id=turn_id, **kw))


def events(bus, etype, turn_id=None):
    return [e for e in bus.events if e.event_type == etype and (turn_id is None or e.turn_id == turn_id)]


def transition(bus, turn_id):
    return events(bus, "answer_version_transition", turn_id)[-1].payload


def hashes(ledger, sub_ids=None):
    return {c.claim_id: c.text_hash for c in ledger.get_verified_claims() if sub_ids is None or c.sub_intent_id in sub_ids}


def test_example_2_refines_instead_of_restarting(index):
    engine, bus, clock = session(index)
    run(engine, clock, TURN1, "t1")
    ledger = engine.ledger
    assert ledger.answer_version == 1 and {s.id for s in ledger.sub_intents} == {"sub_1", "sub_2"}
    v1 = hashes(ledger)
    v1_sub2 = hashes(ledger, {"sub_2"})

    result = run(engine, clock, TURN2, "t2")
    plan = events(bus, "plan_completed", "t2")[-1].payload
    assert plan["relation"] == "modifies" and plan["affected_sub_intents"] == ["sub_1"] and not plan["fallback"]

    # v2: base claims preserved, >= 1 revised or added claim citing the delta evidence.
    assert ledger.answer_version == 2
    t = transition(bus, "t2")
    assert (t["from"], t["to"], t["unchanged_hashes_ok"]) == (1, 2, True)
    assert all(hashes(ledger).get(cid) == h for cid, h in v1.items() if cid not in t["revised"] + t["retracted"])
    assert hashes(ledger, {"sub_2"}) == v1_sub2
    changed = [c for c in ledger.get_verified_claims() if c.claim_id in t["revised"] + t["added"]]
    assert changed and any(DELTA_GOLD in c.doc_ids for c in changed)
    assert ledger.constraints[-1].text == TURN2 and ledger.constraints[-1].affects == ["sub_1"]

    # Rewrite scope: only claims of the affected sub-intent reached the rewriter.
    sub1_v1 = {cid for cid in v1 if cid in hashes(ledger, {"sub_1"})}
    assert set(t["rewrite_input_claim_ids"]) == sub1_v1

    # No full re-search: only delta retrievals (trigger "refinement"), <= 2, fewer than the restart baseline.
    retrievals = events(bus, "speculative_retrieval", "t2")
    assert retrievals and {e.trigger for e in retrievals} == {"refinement"}
    assert result["retrieval_calls"] <= 2
    b1, b1_bus, b1_clock = session(index, mode="b1")
    run(b1, b1_clock, TURN1, "t1")
    restart = run(b1, b1_clock, TURN1, "t2", late_detail=TURN2)
    assert result["retrieval_calls"] < restart["retrieval_calls"]
    assert events(b1_bus, "reconciliation", "t2")[-1].reconciliation_type == "restart"

    # Only added / revised claims are streamed; kept claims are not re-streamed.
    streamed = [e.claim_id for e in events(bus, "answer_chunk", "t2")]
    assert sorted(streamed) == sorted(t["revised"] + t["added"])
    assert events(bus, "answer_chunk", "t2")[0].first_token
    delta = events(bus, "answer_delta", "t2")[-1]
    assert delta.change_type == "refine" and delta.answer_version == 2
    assert {op["op"] for op in delta.ops} <= {"keep", "revise", "retract", "add"}


def revise_first_claim(bad_text=None):
    """Backend that revises the first affected claim with a delta-evidence sentence (or with bad text)."""
    async def fn(prompt, schema, context):
        target = context["claims"][0]
        ev = next(e for e in context["evidence"] if e["chunk_id"] not in target["cites"])
        text = bad_text or split_sentences(ev["text"])[0]
        return json.dumps({"decisions": [{"claim_id": target["claim_id"], "action": "revise", "text": text, "cites": [ev["chunk_id"]]}],
                           "new_claims": []})
    return fn


def test_revision_changes_only_the_revised_claim(index):
    engine, bus, clock = session(index, rewrite_llm=revise_first_claim())
    run(engine, clock, TURN1, "t1")
    before = hashes(engine.ledger)
    run(engine, clock, TURN2, "t2")
    t = transition(bus, "t2")
    assert len(t["revised"]) == 1 and t["unchanged_hashes_ok"]
    after = hashes(engine.ledger)
    assert [cid for cid in before if after[cid] != before[cid]] == t["revised"]
    revised = engine.ledger.get_claim(t["revised"][0])
    assert revised.status == "revised" and revised.history[-1]["version"] == 2
    assert [e.claim_id for e in events(bus, "answer_chunk", "t2")] == t["revised"]


def test_revision_failing_verification_reverts_to_the_previous_text(index):
    engine, bus, clock = session(index, rewrite_llm=revise_first_claim("Dense vectors are stored on the moon in 99 separate warehouses."))
    run(engine, clock, TURN1, "t1")
    before = hashes(engine.ledger)
    run(engine, clock, TURN2, "t2")
    t = transition(bus, "t2")
    assert t["revised"] == [] and len(t["revision_rejected"]) == 1
    assert hashes(engine.ledger) == before
    assert not events(bus, "answer_chunk", "t2")


def test_invalid_planner_json_falls_back_to_adds_and_keeps_v1(index):
    async def broken(prompt, schema, context):
        return "{not json"

    engine, bus, clock = session(index, delta_llm=broken)
    run(engine, clock, TURN1, "t1")
    before = hashes(engine.ledger)
    run(engine, clock, TURN2, "t2")
    plan = events(bus, "plan_completed", "t2")[-1].payload
    assert plan["fallback"] and plan["relation"] == "adds" and plan["delta_queries"] == [TURN2]
    after = hashes(engine.ledger)
    assert all(after.get(cid) == h for cid, h in before.items())  # v1 claims still present, unchanged
    t = transition(bus, "t2")
    assert t["change_type"] == "add" and t["unchanged_hashes_ok"]
    assert not events(bus, "reconciliation", "t2") or all(e.reconciliation_type != "restart" for e in events(bus, "reconciliation", "t2"))


def test_insufficient_delta_evidence_keeps_v1_and_records_uncertainty(index):
    async def off_corpus(prompt, schema, context):
        return json.dumps({"relation": "modifies", "affected_sub_intents": ["sub_1"],
                           "delta_queries": ["martian tax rates for interstellar warp drive engines"]})

    engine, bus, clock = session(index, delta_llm=off_corpus)
    run(engine, clock, TURN1, "t1")
    before, n_claims = hashes(engine.ledger), len(engine.ledger.get_verified_claims())
    run(engine, clock, "Assume martian tax rates apply to interstellar warp drive engines.", "t2")
    assert hashes(engine.ledger) == before and len(engine.ledger.get_verified_claims()) == n_claims
    assert engine.ledger.answer_version == 1
    item = engine.ledger.uncertainty[-1]
    assert item["reason"] == "refinement_insufficient" and item["text"].startswith("could not verify how")
    t = transition(bus, "t2")
    assert (t["from"], t["to"], t["applied"], t["added"]) == (1, 1, False, [])


def test_any_error_restores_the_previous_ledger(index, monkeypatch):
    import slrag.engine.turn_engine as refinement

    def explode(*args, **kwargs):
        raise RuntimeError("simulated failure while applying")

    monkeypatch.setattr(refinement, "apply_delta", explode)
    engine, bus, clock = session(index)
    run(engine, clock, TURN1, "t1")
    before, constraints = hashes(engine.ledger), len(engine.ledger.constraints)
    run(engine, clock, TURN2, "t2")
    assert hashes(engine.ledger) == before and engine.ledger.answer_version == 1
    assert len(engine.ledger.constraints) == constraints  # nothing partially applied
    assert engine.ledger.uncertainty[-1]["reason"] == "refinement_error"
    t = transition(bus, "t2")
    assert (t["from"], t["to"], t["reason"]) == (1, 1, "refinement_error")


def test_presentation_after_refinement_uses_active_claims_and_keeps_the_version(index):
    engine, bus, clock = session(index)
    run(engine, clock, TURN1, "t1")
    run(engine, clock, TURN2, "t2")
    live = {c.claim_id: c for c in engine.ledger.get_verified_claims()}
    result = run(engine, clock, "Please repeat your last answer in two bullets.", "t3")
    assert result.get("suppressed") and result["retrieval_calls"] == 0
    assert engine.ledger.answer_version == 2
    assert not events(bus, "plan_completed", "t3") and not events(bus, "answer_version_transition", "t3")
    cites = {c for cl in live.values() for c in cl.doc_ids}
    assert set(result["turn_output"].citations) <= cites


def test_all_late_detail_scenarios_preserve_unaffected_claims(tmp_path):
    overrides = {"refinement.strict_preservation": True}
    metrics = []
    for split in ("tune", "test"):
        runner = ReplayRunner(str(SCENARIOS / f"late_{split}.jsonl"), mode="ours", out_path=str(tmp_path / f"{split}.jsonl"),
                              config_overrides=overrides)
        result = runner.run(generate_report=False)
        metrics.append(result["metrics"])
        a4 = result["experiments"]["A4"]
        assert a4["ours_retrieval_calls"] < a4["restart_retrieval_calls"]
    assert sum(m["refinement_turns"] for m in metrics) == 15
    for m in metrics:
        assert m["preservation_rate"] == 1.0
        assert m["version_lineage_complete"] == 1.0
        assert m["late_restart_turns"] == 0
        assert m["trace_coverage"] == 1.0
        assert GateEvaluator.g5_late_detail(m)["status"] == "PASS"


def test_refinement_turns_are_excluded_from_the_g2_retrieval_rates():
    from slrag.replay.metrics import MetricCalculator

    events = [
        {"event_type": "speculative_retrieval", "turn_id": "a", "is_early": True, "trigger": "provisional"},
        {"event_type": "speculative_retrieval", "turn_id": "a", "is_early": False, "trigger": "final"},
        # Refinement turn: its retrievals (delta, or Phase 5 after an adds fallback) are judged by G5, not G2.
        {"event_type": "plan_completed", "turn_id": "b", "payload": {"relation": "adds"}},
        {"event_type": "speculative_retrieval", "turn_id": "b", "is_early": False, "trigger": "final"},
        {"event_type": "speculative_retrieval", "turn_id": "c", "is_early": False, "trigger": "refinement"},
        {"event_type": "speculation_outcome", "turn_id": "b", "outcome": "wasted"},
    ]
    m = MetricCalculator.calculate_all(events)
    assert m["early_retrieval_rate"] == 0.5 and m["wasted_speculation_rate"] == 0.0
    assert m["refinement_retrievals"] == 2.0


def test_g5_gate_logic():
    good = {"refinement_turns": 8.0, "preservation_rate": 1.0, "late_restart_turns": 0.0, "version_lineage_complete": 1.0}
    assert GateEvaluator.g5_late_detail(good)["status"] == "PASS"
    assert GateEvaluator.g5_late_detail(dict(good, preservation_rate=0.99))["status"] == "FAIL"
    assert GateEvaluator.g5_late_detail(dict(good, late_restart_turns=1.0))["status"] == "FAIL"
    assert GateEvaluator.g5_late_detail(dict(good, version_lineage_complete=0.5))["status"] == "FAIL"
    assert GateEvaluator.g5_late_detail({"refinement_turns": 0.0})["status"] == "NOT_EVALUATED"
    assert GateEvaluator.g5_late_detail({})["status"] == "FAIL"
