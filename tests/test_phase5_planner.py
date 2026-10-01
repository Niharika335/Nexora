"""Phase 5: multi-intent gate, planner (LLM timeout -> fallback), post-processing, quotas, parallelism, uncertainty."""

import asyncio
import dataclasses
import json
import time
from dataclasses import dataclass
from typing import Any, List

import pytest

from slrag.config import DEFAULT_CONFIG, SlragConfig
from slrag.pipeline.ledger import ClaimLedger
from slrag.pipeline.subintent_executor import SubIntentExecutor
from slrag.pipeline.turn_engine import TurnEngine
from slrag.plan import QueryPlanner, multi_intent_gate
from slrag.retrieval.quota import apply_quota

COMPOUND = [
    "What is Nexora and how does it compare to standard RAG?",
    "I need the cancellation policy and the catering options for the Pune workshop.",
    "Explain the gateway buffering, and what is the dense similarity threshold?",
    "Tell me about Raft and about vector clocks.",
    "How does BM25 work, how does HNSW work, and which is faster?",
    "What is the refund policy? Also, can I bring a guest?",
    "List the telemetry event types and explain how TTFT is measured.",
    "I want the hotel options in Pune plus the train timings from Mumbai.",
    "Check the venue capacity as well as the parking availability.",
    "Describe log compaction and tell me how B+Trees handle range scans.",
]
SINGLE = [
    "What is the dense index dimension?",
    "How does Raft elect a leader?",
    "Explain the token pacing policy of the streaming drafter.",
    "What threshold triggers socket backpressure in the streaming engine?",
    "Summarize the telemetry JSONL serialization format.",
    "Why do LSM trees use a write-ahead log before the MemTable?",
    "Describe graceful degradation when a streaming consumer lags behind the token generator for several seconds.",
    "How is TTFT calculated?",
    "What does the surface code protect against in fault tolerant quantum computers today?",
    "Which component validates ingress packets?",
]


def test_gate_table():
    correct = sum(multi_intent_gate(q).run_planner for q in COMPOUND) + sum(not multi_intent_gate(q).run_planner for q in SINGLE)
    assert correct >= 18, correct


def test_long_single_intent_makes_no_planner_call():
    calls = []

    async def llm(buffer, slots):
        calls.append(buffer)
        return {"sub_queries": [buffer]}

    q = "Could you describe in detail what graceful degradation means when a streaming consumer lags behind the generator for several seconds"
    plan = asyncio.run(QueryPlanner(llm_plan_fn=llm).plan(q))
    assert plan.source == "single" and calls == []


def test_planner_timeout_falls_back_to_splitter():
    async def slow(buffer, slots):
        await asyncio.sleep(1.0)
        return {"sub_queries": [buffer]}

    cfg = dataclasses.replace(DEFAULT_CONFIG.planner, timeout_s=0.001)
    plan = asyncio.run(QueryPlanner(cfg, llm_plan_fn=slow).plan(COMPOUND[1]))
    assert plan.source == "fallback" and plan.plan_fallback == "timeout"
    assert len(plan.sub_queries) >= 2


def test_default_planner_timeout_is_1_5_s():
    assert DEFAULT_CONFIG.planner.timeout_s == 1.5


@pytest.mark.parametrize("raw", ["not json", {"sub_queries": []}, {"sub_queries": ["a"] * 5}, {"other": 1}])
def test_invalid_planner_json_falls_back(raw):
    async def bad(buffer, slots):
        return raw

    plan = asyncio.run(QueryPlanner(llm_plan_fn=bad).plan(COMPOUND[0]))
    assert plan.source == "fallback" and plan.plan_fallback == "invalid_json" and len(plan.sub_queries) == 2


def test_llm_plan_gets_slot_injection_drop_and_merge():
    async def llm(buffer, slots):
        return json.dumps({"sub_queries": [
            "What is the cancellation policy?",
            "What is the CANCELLATION policy?",  # near-duplicate -> merged
            "the catering options",
            "also",  # headless fragment -> dropped
        ]})

    q = "I am planning a workshop in Pune for 30 people and I need the cancellation policy and the catering options."
    plan = asyncio.run(QueryPlanner(llm_plan_fn=llm).plan(q))
    texts = [s.text for s in plan.sub_queries]
    assert plan.source == "llm" and len(texts) == 2
    assert all("Pune" in t and "30" in t for t in texts)
    assert plan.merged and plan.merged[0]["merged"].startswith("What is the CANCELLATION policy?")
    assert plan.dropped and plan.dropped[0]["reason"] == "too_short"


def test_cap_at_four_sub_queries():
    plan = QueryPlanner().plan_sync("I need the hotel, the venue, the catering, the transport, and the agenda for the offsite.")
    assert len(plan.sub_queries) == 4
    assert [d["reason"] for d in plan.dropped] == ["cap"]


def test_quota_per_subquery_and_global_round_robin():
    results = {"a": [f"a{i}" for i in range(10)], "b": ["b0"], "c": [f"c{i}" for i in range(10)], "d": [f"d{i}" for i in range(10)]}
    capped = apply_quota(results, per_sub=4, global_cap=12)
    assert all(len(v) <= 4 for v in capped.values()) and sum(len(v) for v in capped.values()) <= 12
    assert capped["b"] == ["b0"]  # a dominant sub-query does not starve a small one
    assert capped["a"][:3] == ["a0", "a1", "a2"]


@dataclass
class Chunk:
    id: str
    text: str
    score: float = 0.9


class SlowRetrieval:
    def __init__(self):
        self.calls: List[str] = []

    def retrieve(self, query: str) -> List[Chunk]:
        self.calls.append(query)
        time.sleep(0.15)
        return [Chunk(f"{query}#{i}", f"text {i}") for i in range(10)]


class Bus:
    def __init__(self):
        self.events: List[Any] = []

    def publish(self, e: Any) -> None:
        self.events.append(e)


def test_parallel_retrieval_is_concurrent_and_quota_limited():
    retrieval, bus = SlowRetrieval(), Bus()
    executor = SubIntentExecutor(retrieval, bus=bus)
    wave = [{"sub_intent_id": f"sub_{i}", "text": f"query {i}"} for i in range(3)]
    t0 = time.perf_counter()
    evidence = asyncio.run(executor.execute_wave(wave, stage_idx=0, turn_id="t"))
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.4, elapsed  # three 150 ms retrievals overlapped
    assert all(len(v) == 4 for v in evidence.values()) and sum(len(v) for v in evidence.values()) == 12
    events = [e for e in bus.events if e.event_type == "sub_intent_retrieval"]
    assert [len(e.retrieved_chunk_ids) for e in events] == [10, 10, 10]
    assert [len(e.evidence_chunk_ids) for e in events] == [4, 4, 4]


class KeywordRetrieval:
    def retrieve(self, query: str) -> List[Chunk]:
        return [Chunk("doc§1", f"Evidence about {query}")]


class GateRejectingAtlantis:
    def evaluate(self, query, chunks):
        ok = "atlantis" not in query.lower()
        return type("R", (), {"sufficient": ok, "score": 0.9 if ok else 0.1, "reason": "" if ok else "no evidence"})()


class CitingDrafter:
    last_usage = {"tokens_in": 10, "tokens_out": 5, "cost": 0.0}

    def draft(self, query, chunks, temperature=0.0, seed=13):
        return f"Answer to {query} [{chunks[0].id}]."


def test_insufficient_sub_intent_becomes_uncertainty_without_claim():
    bus, ledger = Bus(), ClaimLedger()
    engine = TurnEngine(
        config=SlragConfig(enable_cascade=False, enable_multi_intent=True, enable_verifier=False),
        retrieval_engine=KeywordRetrieval(), llm=None, verifier=None, sufficiency=GateRejectingAtlantis(),
        ledger=ledger, drafter=CitingDrafter(), bus=bus,
    )
    q = "What is the Raft quorum size, how are logs replicated, and what is the ferry schedule to Atlantis?"
    result = asyncio.run(engine.execute_turn(q, turn_id="t"))
    assert (result["resolved_count"], result["suppressed_count"]) == (2, 1)
    assert "Atlantis" in result["turn_output"].uncertainty
    assert len(ledger.get_verified_claims()) == 0 and len(ledger._verified_claims) == 0  # verifier off -> no verified claims
    assert [u["reason"] for u in ledger.uncertainty] == ["insufficient_evidence"]
    assert "Atlantis" not in result["output"]
    assert result["turn_output"].citations == ["doc§1"]
