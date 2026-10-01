"""Phase 8 reconcile: stands / refined / redone decisions and the engine paths behind them."""

import asyncio
from dataclasses import dataclass

import pytest

from slrag.answer.predraft import PendingDraft, SubDraft, VerifiedDraft
from slrag.answer.reconcile import ReconcileOutcome, decide, kept_claims
from slrag.contracts.events import Claim
from slrag.eval.clock import ReplayClock
from slrag.eval.runner import InMemoryBus
from slrag.nlp.lemmas import extract_slots
from slrag.replay.baselines import build_index, build_turn_engine, mode_config
from slrag.retrieval.cache import chunk_id_of, default_embed


@dataclass
class Chunk:
    chunk_id: str
    text: str = ""


def pending_for(subs, evidence, verified=None, text=None):
    text = text or " ".join(t for _, t in subs)
    p = PendingDraft(turn_id="t", committed_text=text, committed_emb=default_embed(text), committed_slots=extract_slots(text), epoch=0)
    for sid, t in subs:
        p.sub_drafts[sid] = SubDraft(sid, t, [Chunk(c) for c in evidence[sid]], sufficient=True,
                                     verified=(verified or {}).get(sid))
    return p


Q = "Explain how Raft leader election and log replication work"


def final(ids):
    return {"sub_1": [Chunk(c) for c in ids]}


def test_identical_evidence_stands():
    d = decide(Q, [("sub_1", Q)], final(["A", "B", "C", "D"]), pending_for([("sub_1", Q)], {"sub_1": ["A", "B", "C", "D"]}))
    assert d.outcome == ReconcileOutcome.STANDS and d.new_chunk_ids == [] and d.reuses_predraft


def test_up_to_two_new_chunks_is_refined():
    d = decide(Q, [("sub_1", Q)], final(["A", "B", "C", "E"]), pending_for([("sub_1", Q)], {"sub_1": ["A", "B", "C", "D"]}))
    assert d.outcome == ReconcileOutcome.REFINED
    assert (d.new_chunk_ids, d.dropped_chunk_ids) == (["E"], ["D"])
    assert d.subs["sub_1"].new_chunk_set == {"E"}


def test_three_new_chunks_is_redone():
    d = decide(Q, [("sub_1", Q)], final(["A", "E", "F", "G"]), pending_for([("sub_1", Q)], {"sub_1": ["A", "B", "C", "D"]}))
    assert d.outcome == ReconcileOutcome.REDONE and d.reason == "3_new_chunks" and not d.reuses_predraft


def test_tail_slot_change_is_redone_even_with_identical_evidence():
    committed = "Book a workshop room for 30 attendees"
    p = pending_for([("sub_1", committed)], {"sub_1": ["A", "B"]}, text=committed)
    d = decide("Book a workshop room for 50 attendees", [("sub_1", "Book a workshop room for 50 attendees")], final(["A", "B"]), p)
    assert (d.outcome, d.reason) == (ReconcileOutcome.REDONE, "slot_conflict")


def test_cancelled_or_unready_predraft_is_redone():
    p = pending_for([("sub_1", Q)], {"sub_1": ["A"]})
    assert decide(Q, [("sub_1", Q)], final(["A"]), p, ready=False).reason == "timeout"
    p.cancel("epoch_changed")
    assert (decide(Q, [("sub_1", Q)], final(["A"]), p).outcome, decide(Q, [("sub_1", Q)], final(["A"]), p).reason) == (
        ReconcileOutcome.REDONE, "epoch_changed")
    assert decide(Q, [("sub_1", Q)], final(["A"]), None).reason == "no_predraft"


def test_validity_is_per_sub_intent():
    s1, s2 = "How does the claim ledger enforce session isolation?", "how does RRF"
    p = pending_for([("sub_1", s1), ("sub_2", s2)], {"sub_1": ["A", "B"], "sub_2": ["X"]})
    final_subs = [("sub_1", s1), ("sub_2", "what telemetry events are emitted on turn start?")]
    d = decide(f"{s1} and what telemetry events are emitted on turn start?", final_subs,
               {"sub_1": [Chunk("A"), Chunk("B")], "sub_2": [Chunk("T")]}, p)
    assert d.outcome == ReconcileOutcome.REFINED and d.redrafted == ["sub_2"]
    assert d.subs["sub_1"].pending is p.sub_drafts["sub_1"] and d.subs["sub_2"].pending is None
    assert [u.sub_intent_id for u in d.unused_predrafts] == ["sub_2"]


def test_kept_claims_drop_those_citing_evidence_that_left():
    claims = [Claim(claim_id="a", text="x.", doc_ids=["A"]), Claim(claim_id="d", text="y.", doc_ids=["D"])]
    sub = SubDraft("sub_1", Q, [Chunk("A"), Chunk("D")], sufficient=True, verified=VerifiedDraft("", claims, 2, 2))
    kept, discarded = kept_claims(sub, ["D"])
    assert [c.claim_id for c in kept] == ["a"] and discarded == 1


# -- engine paths -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def index():
    return build_index(mode_config("ours").app)


class StubRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query):
        return list(self.results)


def run_reconcile(index, predraft_chunks, final_chunks):
    """Pre-draft on `predraft_chunks`, then the utterance_end path on `final_chunks` (cache off)."""
    bus = InMemoryBus()
    engine = build_turn_engine("ours", index, bus, clock=ReplayClock())
    engine.cache = engine.sub_executor.cache = None
    engine.retrieval_engine = engine.sub_executor.retrieval_engine = StubRetriever(final_chunks)
    calls = []
    real_draft = engine.drafter.draft

    def counting_draft(query, chunks, **kw):
        calls.append([chunk_id_of(c) for c in chunks])
        return real_draft(query=query, chunks=chunks, **kw)

    engine.drafter.draft = counting_draft

    async def go():
        async def evidence(sub_id, text):
            return list(predraft_chunks)

        pending = engine.predrafter.start("t", Q, [("sub_1", Q)], evidence)
        await pending.task
        predraft_calls = len(calls)
        state = {"t_start": engine._now(), "t_end": engine._now() + 1.0, "first_emitted": False, "ttft_ms": None}
        engine.clock.advance(1.0)
        totals = {"tokens_in": 0.0, "tokens_out": 0.0, "cost": 0.0, "emitted_cites": 0.0, "hallucinated_cites": 0.0}
        result = await engine._multi_intent_path(Q, "t", [], None, False, state, totals, 0, pending=pending)
        return pending, result, calls[predraft_calls:], totals

    pending, result, final_calls, totals = asyncio.run(go())
    rec = [e for e in bus.events if e.event_type == "reconcile_completed"][-1]
    complete = [e for e in bus.events if e.event_type == "turn_complete"][-1]
    return pending, result, final_calls, rec, complete, totals


@pytest.fixture(scope="module")
def chunks(index):
    ranked = index.search(Q, mode="hybrid", top_k=10)
    assert ranked[0].chunk_id == "DOC002§raft_consensus"
    return ranked[:4], ranked[4:8]


def test_stands_reuses_predraft_claims_with_no_waste(index, chunks):
    base, _ = chunks
    pending, result, final_calls, rec, complete, _ = run_reconcile(index, base, base)
    assert (rec.outcome, rec.wasted_tokens, rec.redone, rec.refined) == ("stands", 0, 0, 0)
    assert rec.stands == len(pending.claims) > 0
    assert final_calls == []  # no drafting at utterance_end
    assert result["output"] == pending.sub_drafts["sub_1"].verified.verified_text
    assert complete.reconcile_outcome == "stands" and complete.wasted_tokens == 0
    assert complete.tokens_in + complete.tokens_out == pending.tokens  # pre-draft tokens counted once


def test_refined_drafts_and_verifies_only_the_new_chunk(index, chunks):
    base, extra = chunks
    final_chunks = base[:3] + [extra[0]]
    pending, result, final_calls, rec, complete, _ = run_reconcile(index, base, final_chunks)
    assert rec.outcome == "refined" and rec.wasted_tokens == 0
    assert rec.new_chunk_ids == [extra[0].chunk_id] and rec.dropped_chunk_ids == [base[3].chunk_id]
    assert final_calls == [[extra[0].chunk_id]]  # partial re-draft / re-verify on the new chunk only
    assert base[3].chunk_id not in result["output"]  # claims citing the dropped chunk are discarded
    assert rec.stands > 0 and rec.discarded_claims > 0


def test_redone_runs_the_full_pipeline_and_counts_waste(index, chunks):
    base, extra = chunks
    final_chunks = [base[0]] + extra[:3]
    pending, result, final_calls, rec, complete, totals = run_reconcile(index, base, final_chunks)
    assert rec.outcome == "redone" and rec.reason == "3_new_chunks"
    assert rec.wasted_tokens == pending.tokens > 0 and complete.wasted_tokens == rec.wasted_tokens
    assert final_calls == [[c.chunk_id for c in final_chunks]]  # full draft on the final evidence
    assert rec.redone > 0 and rec.stands == 0
    # Safety: nothing drafted from the discarded pre-draft evidence reaches the answer.
    assert not {c.chunk_id for c in base[1:]} & set(result["output"].replace("[", " ").replace("]", " ").replace(",", " ").split())
    assert totals["tokens_in"] + totals["tokens_out"] >= pending.tokens  # waste is part of the turn's cost
