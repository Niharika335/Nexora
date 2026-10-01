"""Per-clause early retrieval on multi-intent turns (controller.per_clause_retrieval)."""

import asyncio

import pytest

from slrag.control import RetrievalController, TurnState
from slrag.eval.clock import ReplayClock
from slrag.eval.runner import InMemoryBus
from slrag.replay.baselines import build_index, build_turn_engine, mode_config

COMPOUND = "What does the Growth plan cost per month? and how are API requests authenticated with an API key?"


@pytest.fixture(scope="module")
def index():
    return build_index(mode_config("ours").app)


def run(index, query, overrides=None):
    bus = InMemoryBus()
    engine = build_turn_engine("ours", index, bus, clock=ReplayClock(), overrides=overrides)
    asyncio.run(engine.execute_turn(query, turn_id="t"))
    return bus.events


def test_clause_end_signal():
    async def go(chunks):
        ctl, st, out = RetrievalController(), TurnState(turn_id="t"), []
        for ch in chunks:
            st.chunks.append(ch)
            st.buffer = f"{st.buffer} {ch}".strip()
            out.append(await ctl.on_chunk(st))
        return out, st

    out, st = asyncio.run(go(["What does the", "Growth plan cost", "per month? and", "how are API",
                              "requests authenticated with", "an API key?"]))
    assert st.is_multi_intent
    assert [r.clause_end for r in out] == [False, False, False, False, False, True]  # gate fires at chunk 4
    # A single-intent turn never signals a clause end.
    out, st = asyncio.run(go(["What does the", "Growth plan cost", "per month?"]))
    assert not st.is_multi_intent and not any(r.clause_end for r in out)


def test_each_clause_is_retrieved_before_utterance_end_and_reused(index):
    events = run(index, COMPOUND)
    types = [e.event_type for e in events]
    end = types.index("utterance_final")
    clause = [i for i, e in enumerate(events) if e.event_type == "speculative_retrieval" and e.trigger == "clause"]
    assert clause and all(i < end and events[i].is_early for i in clause)
    second = next(events[i] for i in clause if "authenticated" in events[i].query)
    # The final path finds the completed second clause in the evidence cache: no late retrieval.
    assert not [e for e in events if e.event_type == "speculative_retrieval" and not e.is_early]
    subs = {e.sub_intent_id: e for e in events if e.event_type == "sub_intent_retrieval"}
    assert subs["sub_2"].source == "cache" and "nx-api-auth§api-keys" in subs["sub_2"].retrieved_chunk_ids
    outcome = next(e for e in events if e.event_type == "speculation_outcome" and e.retrieval_id == second.retrieval_id)
    assert outcome.outcome == "used"


def test_disabled_flag_and_single_intent_turns_are_unchanged(index):
    off = run(index, COMPOUND, overrides={"controller.per_clause_retrieval": False})
    assert not [e for e in off if e.event_type == "speculative_retrieval" and e.trigger == "clause"]
    assert [e for e in off if e.event_type == "speculative_retrieval" and not e.is_early]  # the old late retrieval
    single = run(index, "How long does a waitlist offer hold a released ticket before it moves on?")
    assert not [e for e in single if e.event_type == "speculative_retrieval" and e.trigger == "clause"]
