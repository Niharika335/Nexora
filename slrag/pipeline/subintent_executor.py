"""Parallel, cache-aware sub-query retrieval with per-sub-query and global evidence quotas."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from slrag.contracts.events import SubIntentRetrievalEvent
from slrag.eval.clock import DEFAULT_LATENCY, LatencyModel, charge
from slrag.retrieval.cache import chunk_id_of
from slrag.retrieval.quota import apply_quota


def evidence_details(chunks: List[Any]) -> List[Dict[str, Any]]:
    """[{chunk_id, doc_id, section, score}] for telemetry (the trace UI's evidence panel)."""
    out = []
    for i, c in enumerate(chunks):
        score = getattr(c, "score", None)
        out.append({
            "chunk_id": chunk_id_of(c, str(i)), "doc_id": getattr(c, "doc_id", ""),
            "section": getattr(c, "section_title", ""), "score": None if score is None else round(float(score), 4),
        })
    return out


class SubIntentExecutor:
    def __init__(
        self,
        retrieval_engine: Any,
        cache: Optional[Any] = None,
        bus: Optional[Any] = None,
        clock: Optional[Any] = None,
        per_sub_quota: int = 4,
        global_quota: int = 12,
        latency: LatencyModel = DEFAULT_LATENCY,
    ):
        self.retrieval_engine = retrieval_engine
        self.cache = cache
        self.bus = bus
        self.clock = clock
        self.per_sub_quota = per_sub_quota
        self.global_quota = global_quota
        self.latency = latency
        self.retrieval_calls = 0
        self.last_ranked: Dict[str, List[Any]] = {}
        self.last_sources: Dict[str, str] = {}

    async def execute_wave(
        self,
        wave: List[Dict[str, Any]],
        stage_idx: int,
        turn_id: str = "",
        expected_chunk_ids: Optional[List[str]] = None,
        sub_gold_map: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, List[Any]]:
        """Retrieve evidence for every sub-query of a wave concurrently; returns quota-limited evidence by sub id."""
        ranked: Dict[str, List[Any]] = {}
        sources: Dict[str, str] = {}
        wait_until = self.clock.time() if self.clock else 0.0

        # 1. Cache first (evidence cache reuse rule), in wave order.
        misses: List[Dict[str, Any]] = []
        for sub in wave:
            hit = self.cache.lookup(sub["text"], turn_id=turn_id) if self.cache else None
            if hit is not None:
                hit.entry.used = True
                ranked[sub["sub_intent_id"]] = hit.entry.evidence
                sources[sub["sub_intent_id"]] = "cache"
                wait_until = max(wait_until, hit.entry.ready_at)
            else:
                misses.append(sub)

        # 2. Fresh retrievals run concurrently.
        if misses:
            results = await asyncio.gather(*(asyncio.to_thread(self.retrieval_engine.retrieve, s["text"]) for s in misses))
            self.retrieval_calls += len(misses)
            if self.clock:
                wait_until = max(wait_until, self.clock.time() + self.latency.retrieval_s)
            for sub, chunks in zip(misses, results):
                ranked[sub["sub_intent_id"]] = list(chunks)
                sources[sub["sub_intent_id"]] = "fresh"
                if self.cache:
                    self.cache.store(f"{turn_id}:{sub['sub_intent_id']}:final", sub["text"], list(chunks), turn_id=turn_id)
        elif any(src == "cache" for src in sources.values()):
            charge(self.clock, self.latency.cache_hit_s)

        if self.clock is not None and hasattr(self.clock, "advance_to"):
            self.clock.advance_to(wait_until)

        # 3. Quotas across the wave (dedupe within each sub-query only).
        ordered = {sub["sub_intent_id"]: ranked[sub["sub_intent_id"]] for sub in wave}
        evidence = apply_quota(ordered, per_sub=self.per_sub_quota, global_cap=self.global_quota)
        self.last_ranked.update(ordered)
        self.last_sources = dict(sources)

        # 4. Telemetry in wave order.
        if self.bus:
            for sub in wave:
                sub_id = sub["sub_intent_id"]
                if sub_gold_map and sub_id in sub_gold_map:
                    sub_expected = sub_gold_map[sub_id]
                else:
                    sub_expected = expected_chunk_ids or []
                kw: Dict[str, Any] = {
                    "turn_id": turn_id,
                    "sub_intent_id": sub_id,
                    "retrieval_query": sub["text"],
                    "stage_idx": stage_idx,
                    "is_parallel": len(wave) > 1,
                    "source": sources[sub_id],
                    "retrieved_chunk_ids": [chunk_id_of(c, str(i)) for i, c in enumerate(ordered[sub_id])],
                    "evidence_chunk_ids": [chunk_id_of(c, str(i)) for i, c in enumerate(evidence[sub_id])],
                    "expected_chunk_ids": list(sub_expected),
                    "evidence": evidence_details(evidence[sub_id]),
                }
                if self.clock:
                    kw["timestamp"] = self.clock.time()
                self.bus.publish(SubIntentRetrievalEvent(**kw))
        return evidence
