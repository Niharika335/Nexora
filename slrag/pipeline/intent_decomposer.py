"""Multi-intent decomposition: runs the query planner and publishes the resulting sub-intents."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from slrag.contracts.events import SubIntentDecompositionEvent
from slrag.plan.planner import Plan, QueryPlanner

DEPENDENCY_CUE_RE = re.compile(r"^\s*(?:then|after that|based on that|using that)\b", re.IGNORECASE)


class IntentDecomposer:
    def __init__(self, bus: Optional[Any] = None, clock: Optional[Any] = None, planner: Optional[QueryPlanner] = None):
        self.bus = bus
        self.clock = clock
        self.planner = planner or QueryPlanner()
        self.last_plan: Optional[Plan] = None

    def decompose(self, query: str, turn_id: str = "") -> List[Dict[str, Any]]:
        """Synchronous decomposition (no LLM call: gate -> fallback splitter -> post-processing)."""
        return self._publish(query, turn_id, self.planner.plan_sync(query))

    async def adecompose(self, query: str, turn_id: str = "", id_offset: int = 0) -> List[Dict[str, Any]]:
        """Decomposition with the LLM planner (1.5 s timeout, fallback on timeout / invalid JSON).

        `id_offset` numbers the sub-intents after ones already in the session (sub_{offset+1}, ...)."""
        return self._publish(query, turn_id, await self.planner.plan(query), id_offset)

    def _publish(self, query: str, turn_id: str, plan: Plan, id_offset: int = 0) -> List[Dict[str, Any]]:
        self.last_plan = plan
        sub_intents: List[Dict[str, Any]] = []
        for idx, sq in enumerate(plan.sub_queries):
            depends_on = [sub_intents[idx - 1]["sub_intent_id"]] if idx > 0 and DEPENDENCY_CUE_RE.match(sq.text) else []
            sub_intents.append({"sub_intent_id": f"sub_{id_offset + idx + 1}", "text": sq.text, "depends_on": depends_on})

        if self.bus:
            kw: Dict[str, Any] = {
                "turn_id": turn_id,
                "query": query,
                "sub_intents": sub_intents,
                "is_compound": len(sub_intents) > 1,
                "source": plan.source,
                "merged": plan.merged,
                "dropped": plan.dropped,
                "gate_reasons": plan.gate_reasons,
            }
            if self.clock:
                kw["timestamp"] = self.clock.time()
            self.bus.publish(SubIntentDecompositionEvent(**kw))
        return sub_intents
