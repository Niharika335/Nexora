"""Query planner: gate -> (LLM decompose with timeout | fallback splitter) -> post-processing."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

import numpy as np

from slrag.config import DEFAULT_CONFIG, PlannerConfig
from slrag.nlp.lemmas import content_tokens, entities, extract_slots
from slrag.plan.multi_intent_gate import multi_intent_gate, starts_clause
from slrag.plan.splitter import split_fallback
from slrag.retrieval.cache import default_embed

PLANNER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["sub_queries"],
    "properties": {
        "sub_queries": {
            "type": "array", "minItems": 1, "maxItems": 4,
            "items": {"type": "string", "maxLength": 160},
        }
    },
}
SHARED_SLOT_TYPES = ("CARDINAL", "DATE", "MONEY", "PERCENT")

PlanFn = Callable[[str, Dict[str, str]], Awaitable[Any]]


@dataclass
class SubQuery:
    id: str
    text: str


@dataclass
class Plan:
    sub_queries: List[SubQuery]
    source: str  # llm | fallback | single
    shared_slots: Dict[str, str] = field(default_factory=dict)
    dropped: List[Dict[str, Any]] = field(default_factory=list)
    merged: List[Dict[str, Any]] = field(default_factory=list)
    gate_reasons: List[str] = field(default_factory=list)
    plan_fallback: Optional[str] = None  # why the LLM path was not used
    latency_ms: float = 0.0


def validate_planner_output(raw: Any) -> List[str]:
    """Check an LLM response against PLANNER_SCHEMA; raise ValueError if it does not conform."""
    data = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(data, dict) or not isinstance(data.get("sub_queries"), list):
        raise ValueError("planner output must be an object with a sub_queries array")
    subs = data["sub_queries"]
    if not 1 <= len(subs) <= 4:
        raise ValueError("sub_queries must have 1..4 items")
    if not all(isinstance(s, str) and s.strip() and len(s) <= 160 for s in subs):
        raise ValueError("each sub_query must be a non-empty string of <= 160 chars")
    return [s.strip() for s in subs]


def shared_slots_for(buffer: str, preamble: str = "") -> Dict[str, str]:
    """Context every sub-query must carry: quantities/dates anywhere, plus entities from the preamble."""
    slots = {k: v for k, v in extract_slots(buffer).items() if k in SHARED_SLOT_TYPES}
    preamble_entities = entities(preamble) if preamble else []
    if preamble_entities:
        slots["ENTITY"] = "|".join(sorted(preamble_entities))
    return slots


class QueryPlanner:
    def __init__(
        self,
        config: PlannerConfig = DEFAULT_CONFIG.planner,
        llm_plan_fn: Optional[PlanFn] = None,
        embed: Callable[[str], np.ndarray] = default_embed,
        use_spacy: bool = True,
    ):
        self.cfg = config
        self.llm_plan_fn = llm_plan_fn
        self.embed = embed
        self.use_spacy = use_spacy

    async def plan(self, buffer: str, timeout_s: Optional[float] = None) -> Plan:
        """Decompose `buffer`. The LLM call is bounded by timeout_s (default planner.timeout_s = 1.5 s);
        on timeout, exception or invalid JSON the fallback splitter is used (source = "fallback")."""
        t0 = time.perf_counter()
        gate = multi_intent_gate(buffer)
        if not gate.run_planner:
            return self._single(buffer, gate.reasons, t0)

        split = split_fallback(buffer, use_spacy=self.use_spacy)
        slots = shared_slots_for(buffer, split.preamble)
        texts: Optional[List[str]] = None
        fallback_reason: Optional[str] = "no_llm"
        if self.llm_plan_fn is not None:
            try:
                raw = await asyncio.wait_for(self.llm_plan_fn(buffer, slots), timeout=timeout_s or self.cfg.timeout_s)
                texts = validate_planner_output(raw)
                fallback_reason = None
            except asyncio.TimeoutError:
                fallback_reason = "timeout"
            except (ValueError, json.JSONDecodeError):
                fallback_reason = "invalid_json"
            except Exception:
                fallback_reason = "llm_error"

        source = "llm" if texts is not None else "fallback"
        plan = self._post_process(texts if texts is not None else split.parts, slots, buffer, source)
        plan.gate_reasons, plan.plan_fallback = gate.reasons, fallback_reason
        plan.latency_ms = (time.perf_counter() - t0) * 1000.0
        return plan

    def plan_sync(self, buffer: str) -> Plan:
        """Planner without the LLM call (gate -> fallback splitter -> post-processing)."""
        t0 = time.perf_counter()
        gate = multi_intent_gate(buffer)
        if not gate.run_planner:
            return self._single(buffer, gate.reasons, t0)
        split = split_fallback(buffer, use_spacy=self.use_spacy)
        plan = self._post_process(split.parts, shared_slots_for(buffer, split.preamble), buffer, "fallback")
        plan.gate_reasons, plan.plan_fallback = gate.reasons, "no_llm"
        plan.latency_ms = (time.perf_counter() - t0) * 1000.0
        return plan

    def _single(self, buffer: str, reasons: List[str], t0: float) -> Plan:
        return Plan(
            sub_queries=[SubQuery("q1", buffer.strip())], source="single", gate_reasons=reasons,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def _post_process(self, texts: List[str], shared_slots: Dict[str, str], buffer: str, source: str) -> Plan:
        dropped: List[Dict[str, Any]] = []
        merged: List[Dict[str, Any]] = []

        # 1. Slot injection: every sub-query carries every shared slot value.
        injected = []
        for text in texts:
            for value in (v for vals in shared_slots.values() for v in vals.split("|")):
                if value and value.lower() not in text.lower():
                    text = f"{text} {value}"
            injected.append(text)

        # 2. Drop headless fragments with too few content tokens. Whole question/request
        #    clauses ("What is Nexora?") are kept: they are complete, searchable sub-intents.
        kept: List[str] = []
        for text in injected:
            if len(content_tokens(text)) < self.cfg.min_content_tokens and not starts_clause(text):
                dropped.append({"text": text, "reason": "too_short"})
            else:
                kept.append(text)

        # 3. Merge near-duplicates (over-fragmentation guard): keep the earlier one.
        embs = [self.embed(t) for t in kept]
        survivors: List[int] = []
        for j in range(len(kept)):
            match = next((i for i in survivors if float(np.dot(embs[i], embs[j])) >= self.cfg.merge_cos), None)
            if match is None:
                survivors.append(j)
            else:
                merged.append({"kept": kept[match], "merged": kept[j], "cos": round(float(np.dot(embs[match], embs[j])), 4)})
        kept = [kept[i] for i in survivors]

        # 4. Cap, keeping utterance order.
        for text in kept[self.cfg.max_sub_queries:]:
            dropped.append({"text": text, "reason": "cap"})
        kept = kept[: self.cfg.max_sub_queries]

        # 5. Nothing left -> single-query plan.
        if not kept:
            kept, source = [buffer.strip()], "single"

        return Plan(
            sub_queries=[SubQuery(f"q{i + 1}", t) for i, t in enumerate(kept)],
            source=source, shared_slots=shared_slots, dropped=dropped, merged=merged,
        )
