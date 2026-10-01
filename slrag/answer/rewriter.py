"""Rewriter: keep | revise | retract per affected claim, plus new claims (Phase 7).

One json_call whose schema enumerates the affected claim ids, the affected sub-intent ids and
the allowed cites = cites of the affected claims ∪ new delta evidence ∪ evidence already held
for the affected sub-intents. Only claims of affected sub-intents are sent: anything else
raises RewriteScopeError before the call, so unaffected claims can never be rewritten.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, Iterable, List, Optional

from slrag.contracts.events import Claim
from slrag.llm.heuristic import heuristic_rewrite_llm
from slrag.llm.json_call import JsonCallError, JsonFn, json_call
from slrag.llm.prompts import rewriter_prompt
from slrag.llm.schemas import rewriter_schema
from slrag.retrieval.cache import chunk_id_of


class RewriteScopeError(ValueError):
    """A claim outside the affected sub-intents was about to reach the rewrite LLM."""


@dataclass
class RewriteDecision:
    claim_id: str
    action: str  # keep | revise | retract
    text: Optional[str] = None
    cites: Optional[List[str]] = None


@dataclass
class NewClaimSpec:
    sub_intent_id: str
    text: str
    cites: List[str]


@dataclass
class RewriteResult:
    decisions: List[RewriteDecision]
    new_claims: List[NewClaimSpec]
    input_claim_ids: List[str]
    allowed_cites: List[str]
    fallback: bool = False
    fallback_reason: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    raw: str = ""
    evidence_sent: Dict[str, List[str]] = field(default_factory=dict)


def allowed_cites(
    affected_claims: Iterable[Claim],
    delta_evidence: Dict[str, List[Any]],
    held_evidence: Dict[str, Iterable[str]],
) -> List[str]:
    cites: List[str] = []
    for c in affected_claims:
        cites.extend(c.doc_ids)
    for chunks in delta_evidence.values():
        cites.extend(chunk_id_of(ch, "") for ch in chunks)
    for ids in held_evidence.values():
        cites.extend(ids)
    return sorted({c for c in cites if c})


class Rewriter:
    def __init__(
        self,
        llm_fn: Optional[JsonFn] = heuristic_rewrite_llm,
        timeout_s: float = 1.5,
        max_new_claims: int = 3,
        evidence_per_sub_intent: int = 4,
    ):
        self.llm_fn = llm_fn
        self.timeout_s = timeout_s
        self.max_new_claims = max_new_claims
        self.evidence_per_sub_intent = evidence_per_sub_intent

    async def rewrite(
        self,
        constraint: str,
        affected_sub_intents: List[str],
        affected_claims: List[Claim],
        delta_evidence: Dict[str, List[Any]],
        held_evidence: Optional[Dict[str, Iterable[str]]] = None,
    ) -> RewriteResult:
        t0 = time.perf_counter()
        outside = [c.claim_id for c in affected_claims if c.sub_intent_id not in affected_sub_intents]
        if outside:
            raise RewriteScopeError(f"claims {outside} are not in the affected sub-intents {affected_sub_intents}")

        held = {sid: list(ids) for sid, ids in (held_evidence or {}).items() if sid in affected_sub_intents}
        evidence = {sid: list(chunks)[: self.evidence_per_sub_intent] for sid, chunks in delta_evidence.items() if sid in affected_sub_intents}
        cites = allowed_cites(affected_claims, evidence, held)
        claim_ids = [c.claim_id for c in affected_claims]
        schema = rewriter_schema(claim_ids, cites, affected_sub_intents, self.max_new_claims)

        claims_in = [{"claim_id": c.claim_id, "sub_intent_id": c.sub_intent_id, "text": c.text, "cites": list(c.doc_ids)} for c in affected_claims]
        evidence_in = [
            {"chunk_id": chunk_id_of(ch, ""), "sub_intent_id": sid, "text": getattr(ch, "text", "")}
            for sid, chunks in evidence.items() for ch in chunks
        ]
        base = RewriteResult(
            decisions=[], new_claims=[], input_claim_ids=claim_ids, allowed_cites=cites,
            evidence_sent={sid: [chunk_id_of(ch, "") for ch in chunks] for sid, chunks in evidence.items()},
        )
        if self.llm_fn is None:
            return self._keep_all(base, "no_llm", t0)

        prompt = rewriter_prompt(constraint, claims_in, evidence_in, self.max_new_claims)
        context = {"constraint": constraint, "claims": claims_in, "evidence": evidence_in, "allowed_cites": cites,
                   "max_new_claims": self.max_new_claims}
        try:
            res = await json_call(self.llm_fn, prompt, schema, context, timeout_s=self.timeout_s)
        except JsonCallError as exc:
            base.tokens_in = exc.tokens_in
            return self._keep_all(base, exc.kind, t0)

        by_claim = {c.claim_id: c for c in affected_claims}
        decisions: Dict[str, RewriteDecision] = {}
        for d in res.data["decisions"]:
            if d["claim_id"] in decisions:
                base.tokens_in, base.tokens_out = res.tokens_in, res.tokens_out
                return self._keep_all(base, "duplicate_decision", t0)
            if d["action"] == "revise" and not (d.get("text") or "").strip():
                base.tokens_in, base.tokens_out = res.tokens_in, res.tokens_out
                return self._keep_all(base, "revise_without_text", t0)
            decisions[d["claim_id"]] = RewriteDecision(
                d["claim_id"], d["action"], (d.get("text") or "").strip() or None,
                list(d["cites"]) if d.get("cites") else (list(by_claim[d["claim_id"]].doc_ids) if d["action"] == "revise" else None),
            )
        for cid in claim_ids:  # a claim the model did not mention is kept
            decisions.setdefault(cid, RewriteDecision(cid, "keep"))

        base.decisions = [decisions[cid] for cid in claim_ids]
        base.new_claims = [NewClaimSpec(n["sub_intent_id"], n["text"].strip(), list(n["cites"])) for n in res.data["new_claims"] if n["text"].strip()]
        base.tokens_in, base.tokens_out, base.raw = res.tokens_in, res.tokens_out, res.raw
        base.latency_ms = (time.perf_counter() - t0) * 1000.0
        return base

    @staticmethod
    def _keep_all(base: RewriteResult, reason: str, t0: float) -> RewriteResult:
        base.decisions = [RewriteDecision(cid, "keep") for cid in base.input_claim_ids]
        base.new_claims = []
        base.fallback, base.fallback_reason = True, reason
        base.latency_ms = (time.perf_counter() - t0) * 1000.0
        return base
