"""Delta planner: how does a later utterance relate to the existing answer? (Phase 7)

One json_call (default timeout 1.5 s) with an enum-constrained schema returns
{relation: modifies|adds|unrelated, affected_sub_intents: [current ids], delta_queries: <= 2}.
On timeout, invalid JSON, a schema violation or `modifies` without affected sub-intents the
fallback applies: relation = "adds" with the utterance itself as the single query, logged as
`delta_fallback`. The session is never restarted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Dict, List, Optional

from slrag.llm.heuristic import heuristic_delta_llm
from slrag.llm.json_call import JsonCallError, JsonFn, json_call
from slrag.llm.prompts import delta_planner_prompt, first_words
from slrag.llm.schemas import delta_planner_schema

logger = logging.getLogger(__name__)


@dataclass
class DeltaPlan:
    relation: str  # modifies | adds | unrelated
    affected_sub_intents: List[str]
    delta_queries: List[str]
    fallback: bool = False
    fallback_reason: Optional[str] = None  # timeout | invalid_json | schema | llm_error | empty_affected
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    raw: str = ""
    input_sub_intents: List[Dict[str, Any]] = field(default_factory=list)

    def payload(self) -> Dict[str, Any]:
        return {
            "relation": self.relation,
            "affected_sub_intents": list(self.affected_sub_intents),
            "delta_queries": list(self.delta_queries),
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
        }


def planner_inputs(ledger: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Sub-intents [{id, text}] and claim one-liners [{claim_id, sub_intent_id, first_20_words, cites}] from a
    ledger. The cites (doc§section ids) tell the planner which document each claim came from."""
    return {
        "sub_intents": [{"id": s.id, "text": s.text} for s in ledger.sub_intents],
        "claims": [
            {"claim_id": c.claim_id, "sub_intent_id": c.sub_intent_id, "first_20_words": first_words(c.text),
             "cites": list(c.doc_ids)}
            for c in ledger.get_verified_claims()
        ],
    }


class DeltaPlanner:
    def __init__(self, llm_fn: Optional[JsonFn] = heuristic_delta_llm, timeout_s: float = 1.5, max_queries: int = 2):
        self.llm_fn = llm_fn
        self.timeout_s = timeout_s
        self.max_queries = max_queries

    def _fallback(self, utterance: str, reason: str, t0: float, tokens_in: int = 0, raw: str = "") -> DeltaPlan:
        logger.warning("delta_fallback: %s", reason)
        return DeltaPlan(
            relation="adds", affected_sub_intents=[], delta_queries=[utterance.strip()[:160]],
            fallback=True, fallback_reason=reason, tokens_in=tokens_in, raw=raw,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    async def plan(self, utterance: str, sub_intents: List[Dict[str, Any]], claims: List[Dict[str, Any]]) -> DeltaPlan:
        t0 = time.perf_counter()
        ids = [s["id"] for s in sub_intents]
        schema = delta_planner_schema(ids, self.max_queries)
        if self.llm_fn is None:
            return self._fallback(utterance, "no_llm", t0)

        prompt = delta_planner_prompt(utterance, sub_intents, claims)
        context = {"utterance": utterance, "sub_intents": sub_intents, "claims": claims, "max_queries": self.max_queries}
        try:
            res = await json_call(self.llm_fn, prompt, schema, context, timeout_s=self.timeout_s)
        except JsonCallError as exc:
            return self._fallback(utterance, exc.kind, t0, tokens_in=exc.tokens_in)

        data = res.data
        affected = list(dict.fromkeys(data["affected_sub_intents"]))  # enum-checked: every id exists
        queries = [q.strip() for q in data["delta_queries"] if q.strip()]
        if data["relation"] == "modifies" and not affected:
            return self._fallback(utterance, "empty_affected", t0, tokens_in=res.tokens_in, raw=res.raw)
        if data["relation"] != "modifies":
            affected = []  # adds / unrelated take the Phase 5 flow for the new content only
        return DeltaPlan(
            relation=data["relation"], affected_sub_intents=affected, delta_queries=queries or [utterance.strip()[:160]],
            tokens_in=res.tokens_in, tokens_out=res.tokens_out, latency_ms=(time.perf_counter() - t0) * 1000.0,
            raw=res.raw, input_sub_intents=sub_intents,
        )
