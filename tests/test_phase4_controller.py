"""Phase 4: T0/T1/T2 controller, evidence cache reuse rule, streaming epochs, Presenter and suppression."""

import asyncio
import dataclasses
from typing import Any, List

import numpy as np
import pytest

from slrag.config import DEFAULT_CONFIG, ControllerConfig
from slrag.contracts.events import Claim, ClaimStatus, SpeculationOutcomeEvent, SpeculativeRetrievalEvent
from slrag.control import Decision, RetrievalController, T2Classifier, TurnState
from slrag.eval.clock import ReplayClock
from slrag.pipeline.ledger import ClaimLedger
from slrag.pipeline.presenter import Presenter, parse_bullet_count
from slrag.pipeline.streaming import StreamingTurn
from slrag.replay.baselines import build_index, build_turn_engine
from slrag.retrieval.cache import EvidenceCache


class Bus:
    def __init__(self):
        self.events: List[Any] = []

    def publish(self, e: Any) -> None:
        self.events.append(e)


def run_stream(chunks, ledger=None, config=DEFAULT_CONFIG.controller, t2=None):
    async def go():
        ctl, st, out = RetrievalController(config, t2=t2), TurnState(turn_id="t"), []
        for ch in chunks:
            st.buffer = f"{st.buffer} {ch}".strip()
            out.append(await ctl.on_chunk(st, ledger))
        return out
    return asyncio.run(go())


def ledger_with_claims() -> ClaimLedger:
    ledger = ClaimLedger("s")
    ledger.add_verified_claims([
        Claim(claim_id="c1", text="A Raft cluster typically has five nodes and tolerates two failures.", doc_ids=["DOC002§raft_consensus"], verification_status=ClaimStatus.VERIFIED),
        Claim(claim_id="c2", text="The leader replicates log entries to a majority quorum.", doc_ids=["DOC002§raft_consensus"], verification_status=ClaimStatus.VERIFIED),
        Claim(claim_id="c3", text="Vector clocks detect causality violations.", doc_ids=["DOC002§vector_clocks"], verification_status=ClaimStatus.VERIFIED),
    ], turn_id="t0")
    return ledger


def test_decision_table_workshop_stream():
    r = run_stream(["I need to plan a customer workshop in…", "…Pune for 30 people, and I need…", "…the cancellation policy and the catering options."])
    assert r[0].decision == Decision.WAIT
    assert r[1].decision == Decision.PROVISIONAL and "Pune" in r[1].query and "30" in r[1].query
    assert r[2].decision == Decision.COMMIT and r[2].reason == "multi_intent_boundary"
    assert r[2].commit_hint["multi_intent"] is True


def test_filler_stream_never_retrieves():
    assert all(r.decision == Decision.WAIT for r in run_stream(["um so", "well the", "uh okay"]))


def test_t0_suppresses_presentation_request():
    r = run_stream(["Please repeat your last answer in two bullets."], ledger_with_claims())
    assert (r[0].decision, r[0].tier, r[0].reason) == (Decision.SUPPRESS, "T0", "presentation_restructure")


def test_presentation_verb_with_new_anchor_is_not_suppressed():
    r = run_stream(["Summarize the policy for international trips to Tokyo"], ledger_with_claims())
    assert r[0].decision != Decision.SUPPRESS


def test_t0_needs_a_ledger():
    assert run_stream(["Please repeat your last answer in two bullets."])[0].decision != Decision.SUPPRESS


def test_t2_timeout_falls_back_to_retrieve():
    async def slow(_):
        await asyncio.sleep(1.0)
        return "NO_RETRIEVE"
    assert asyncio.run(T2Classifier(slow, timeout_s=0.01).decide("x")) == ("RETRIEVE", "t2_timeout")


def test_llm_only_mode_suppresses_on_no_retrieve():
    async def no(_):
        return "NO_RETRIEVE"
    cfg = dataclasses.replace(DEFAULT_CONFIG.controller, mode="llm_only")
    r = run_stream(["Raft leader election with five nodes and two failures"], config=cfg, t2=T2Classifier(no))
    assert (r[0].decision, r[0].tier) == (Decision.SUPPRESS, "T2")


def test_eager_and_batch_modes():
    chunks = ["What is Raft", "leader election", "with 5 nodes"]
    eager = run_stream(chunks, config=dataclasses.replace(DEFAULT_CONFIG.controller, mode="eager"))
    assert [r.decision for r in eager] == [Decision.PROVISIONAL] * 3
    batch = run_stream(chunks, config=dataclasses.replace(DEFAULT_CONFIG.controller, mode="batch"))
    assert all(r.decision == Decision.WAIT for r in batch)


def test_cache_same_query_hits_and_slot_conflict_misses():
    cache = EvidenceCache(ttl_ms=None)
    cache.store("r1", "hotel options in Pune for 30 people", ["e1"])
    assert cache.lookup("hotel options in Pune for 30 people").entry.retrieval_id == "r1"
    assert cache.lookup("hotel options in Pune for 50 people") is None  # CARDINAL 30 vs 50


def test_cache_cosine_threshold():
    base = np.zeros(4, dtype=np.float32)
    base[0] = 1.0

    def vec(cos):
        v = np.array([cos, np.sqrt(1 - cos * cos), 0, 0], dtype=np.float32)
        return v

    embeds = {"stored": base, "q80": vec(0.80), "q84": vec(0.84)}
    cache = EvidenceCache(ttl_ms=None, embed=lambda q: embeds[q])
    cache.store("r", "stored", ["e"])
    assert cache.lookup("q80") is None
    assert cache.lookup("q84") is not None


def test_presenter_bullets_cites_and_fallback():
    ledger = ledger_with_claims()
    assert parse_bullet_count("in two bullets") == 2 and parse_bullet_count("as 4 points") == 4 and parse_bullet_count("shorter") == 3
    res = asyncio.run(Presenter().render("Please repeat your last answer in two bullets.", ledger))
    assert len(res.bullets) == 2 and res.fallback
    assert {c for b in res.bullets for c in b["cites"]} <= {c for cl in ledger.get_verified_claims() for c in cl.doc_ids}

    async def invents_number(instruction, claims, n):
        return {"bullets": [{"text": "Raft needs 9 nodes.", "claim_ids": ["c1"]}]}
    bad = asyncio.run(Presenter(invents_number).render("two bullets", ledger))
    assert bad.fallback and bad.reason == "post_check_failed" and "unseen_number" in bad.failed_checks

    async def faithful(instruction, claims, n):
        return {"bullets": [{"text": "A Raft cluster typically has five nodes.", "claim_ids": ["c1"]}]}
    good = asyncio.run(Presenter(faithful).render("one bullet", ledger))
    assert not good.fallback and good.bullets[0]["cites"] == ["DOC002§raft_consensus"]


def test_suppression_turn_end_to_end_has_zero_retrievals():
    bus, clock = Bus(), ReplayClock()
    index = build_index(DEFAULT_CONFIG)
    engine = build_turn_engine("ours", index, bus, clock=clock)
    asyncio.run(engine.execute_turn("How does Raft leader election and log replication work?", turn_id="t1"))
    version = engine.ledger.answer_version
    cited_before = {c for cl in engine.ledger.get_verified_claims() for c in cl.doc_ids}
    assert version == 1 and cited_before

    result = asyncio.run(engine.execute_turn("Please repeat your last answer in two bullets.", turn_id="t2"))
    t2_events = [e for e in bus.events if getattr(e, "turn_id", "") == "t2"]
    assert result["retrieval_calls"] == 0
    assert not [e for e in t2_events if isinstance(e, SpeculativeRetrievalEvent)]
    assert result["turn_output"].retrieval_events == []
    assert result["turn_output"].meta["reason"] == "presentation_restructure"
    assert result["output"].count("\n- ") + result["output"].startswith("- ") == 2
    assert set(result["turn_output"].citations) <= cited_before
    assert engine.ledger.answer_version == version


def test_early_retrieval_starts_before_utterance_end():
    bus, clock = Bus(), ReplayClock()
    engine = build_turn_engine("ours", build_index(DEFAULT_CONFIG), bus, clock=clock)
    asyncio.run(engine.execute_turn("Explain how Raft leader election tolerates two failures in a five node cluster", turn_id="t"))
    early = [e for e in bus.events if isinstance(e, SpeculativeRetrievalEvent) and e.is_early]
    utt_end = next(e for e in bus.events if e.event_type == "utterance_final")
    assert early and early[0].timestamp < utt_end.timestamp


def test_stale_provisional_result_is_discarded():
    bus, clock = Bus(), ReplayClock()

    async def go():
        turn = StreamingTurn("t", RetrievalController(), lambda q: [f"evidence for {q}"], EvidenceCache(ttl_ms=None, clock=clock),
                             bus=bus, clock=clock)
        first = await turn.on_chunk("Book the Pune hotel for 30 people")
        second = await turn.on_chunk("near the Mumbai airport with 50 rooms")  # new information before the first lands
        await turn.settle()
        return first, second, turn

    first, second, turn = asyncio.run(go())
    assert first.decision == Decision.PROVISIONAL and second.reason == "new_information"
    assert turn.state.epoch == 1 and turn.dispatches[0].stale and not turn.dispatches[1].stale
    stale = [e for e in bus.events if isinstance(e, SpeculationOutcomeEvent) and e.outcome == "stale_discard"]
    assert [e.retrieval_id for e in stale] == [turn.dispatches[0].retrieval_id]
    assert turn.cache.lookup(turn.dispatches[1].query) is not None
