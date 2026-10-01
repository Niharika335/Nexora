"""Refinement routing for later turns of a session (Phase 7: refine, don't restart).

The replay/streaming TurnEngine (slrag.pipeline.turn_engine) hands a non-suppressed turn to
RefinementRouter.run when the session ledger already has an answer (answer_version >= 1):

  delta planner (one json_call) -> plan_completed
    modifies      -> constraint record -> delta retrieval (trigger "refinement", cache first,
                     <= 2 queries) -> sufficiency gate per affected sub-intent -> rewriter (only
                     affected claims) -> verify revised/new claims -> contradiction guard ->
                     apply_delta (answer_version + 1, preservation assertion)
    adds/unrelated -> Phase 5 flow for the new content only, new sub-intent ids appended;
                     existing claims untouched
  -> answer_delta (ops) + answer_chunk for added/revised claims only -> answer_version_transition

Fail-safe: any error restores the ledger checkpoint taken before the turn (nothing is ever
partially applied), records an uncertainty item and still emits the transition event.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from slrag.answer.contradiction import find_contradiction, guard
from slrag.answer.rewriter import Rewriter
from slrag.config import RefinementConfig
from slrag.contracts.events import (
    AnswerChunk,
    AnswerDelta,
    AnswerVersionTransitionEvent,
    Claim,
    Constraint,
    PlanCompletedEvent,
    SpeculativeRetrievalEvent,
    SubIntentCompletionEvent,
    SubIntentRetrievalEvent,
    TurnOutput,
    compute_text_hash,
)
from slrag.eval.clock import charge
from slrag.llm.heuristic import heuristic_delta_llm, heuristic_rewrite_llm
from slrag.llm.json_call import JsonFn
from slrag.nlp.lemmas import content_lemmas
from slrag.pipeline.llm import LLMServiceWrapper
from slrag.pipeline.subintent_executor import evidence_details
from slrag.pipeline.suppression import gate_fields, normalize_gate_result
from slrag.plan.delta_planner import DeltaPlan, DeltaPlanner, planner_inputs
from slrag.retrieval.cache import chunk_id_of
from slrag.state.delta import Decision, NewClaim, PreservationError, apply_delta

logger = logging.getLogger(__name__)


def form_delta_query(planner_query: str, sub_query: str) -> str:
    """The planner's query with the affected sub-query text appended if it is not already covered."""
    if content_lemmas(sub_query) <= content_lemmas(planner_query):
        return planner_query
    return f"{planner_query} {sub_query}".strip()


def conflicts_in(claims: List[Claim]) -> int:
    """Pairs of live claims that the contradiction guard would flag (post-refinement audit)."""
    return sum(1 for i, c in enumerate(claims) if find_contradiction(c.text, claims[:i] + claims[i + 1:]) is not None)


class RefinementRouter:
    def __init__(
        self,
        engine: Any,
        cfg: RefinementConfig,
        delta_llm: Optional[JsonFn] = heuristic_delta_llm,
        rewrite_llm: Optional[JsonFn] = heuristic_rewrite_llm,
    ):
        self.engine = engine
        self.cfg = cfg
        self.planner = DeltaPlanner(delta_llm, timeout_s=cfg.planner_timeout_s, max_queries=cfg.max_delta_queries)
        self.rewriter = Rewriter(
            rewrite_llm, timeout_s=cfg.rewriter_timeout_s, max_new_claims=cfg.max_new_claims,
            evidence_per_sub_intent=cfg.evidence_per_sub_intent,
        )
        self.pricing = LLMServiceWrapper(engine.config.app.llm)
        self.last_rewrite: Optional[Any] = None
        self.last_plan: Optional[DeltaPlan] = None

    def should_route(self) -> bool:
        ledger = self.engine.ledger
        return bool(self.cfg.enabled and ledger is not None and hasattr(ledger, "checkpoint")
                    and getattr(ledger, "answer_version", 0) >= 1)

    # -- helpers ---------------------------------------------------------------------

    def _charge_tokens(self, totals: Dict[str, float], tokens_in: int, tokens_out: int) -> None:
        if tokens_in:  # a json_call was made (a no-LLM fallback costs nothing)
            totals["llm_calls"] = totals.get("llm_calls", 0) + 1
        totals["tokens_in"] += tokens_in
        totals["tokens_out"] += tokens_out
        totals["cost"] += self.pricing.calculate_cost(tokens_in, tokens_out)

    def _transition(self, turn_id: str, payload: Dict[str, Any]) -> None:
        self.engine.record_transition(turn_id, payload)

    def _verify(self, claims: List[Claim], chunks: List[Any], previous: List[Claim], turn_id: str) -> Tuple[set, Optional[float]]:
        """Phase 3 verifier on revised/new claims only. Returns (passing claim ids, groundedness)."""
        e = self.engine
        if not claims:
            return set(), None
        if not (e.config.enable_verifier and e.verifier):
            return {c.claim_id for c in claims}, None
        if hasattr(e.verifier, "verify_claims"):
            res = e.verifier.verify_claims(claims, chunks, turn_id=turn_id, previous=previous)
            return set(res.supported), float(res.groundedness_score)
        passed = set()
        for c in claims:
            text, _ = e._verify(f"{c.text.rstrip('.')} [{', '.join(c.doc_ids)}].", chunks, turn_id)
            if text:
                passed.add(c.claim_id)
        return passed, len(passed) / len(claims)

    def _no_change(
        self, turn_id: str, utterance: str, plan: DeltaPlan, state: Dict[str, Any], totals: Dict[str, float],
        retrieval_calls: int, uncertainty: List[Dict[str, Any]], reason: str, **extra: Any,
    ) -> Dict[str, Any]:
        """Keep every claim and the version unchanged; record uncertainty items; still emit the transition."""
        e = self.engine
        ledger = e.ledger
        for item in uncertainty:
            ledger.add_uncertainty(**item)
        n = ledger.answer_version
        live = ledger.get_verified_claims()
        self._transition(turn_id, {
            "from": n, "to": n, "kept": [c.claim_id for c in live], "revised": [], "retracted": [], "added": [],
            "unchanged_hashes_ok": True, "rewrite_input_claim_ids": extra.pop("rewrite_input_claim_ids", []),
            "unaffected": [], "preserved_checked": 0, "preserved_identical": 0,
            "applied": False, "reason": reason, "change_type": "refine" if plan.relation == "modifies" else "add",
            "relation": plan.relation, **extra,
        })
        lines = [u["text"] for u in uncertainty if u.get("text")]
        message = " ".join(lines) or "The previous answer still applies."
        e._publish(AnswerDelta(turn_id=turn_id, text_delta=message, is_final=True,
                               ops=[{"op": "keep", "claim_id": c.claim_id} for c in live],
                               change_type="refine", rendered_answer=ledger.render(), answer_version=n,
                               hashes={c.claim_id: c.text_hash for c in live}))
        e._first_token(turn_id, message, state)
        e._complete(turn_id, message, state, totals, retrieval_calls)
        return {
            "output": message, "refined": False, "relation": plan.relation, "retrieval_calls": retrieval_calls,
            "latency_ms": (e._now() - state["t_start"]) * 1000.0, "resolved_count": 0, "suppressed_count": 0,
            "turn_output": TurnOutput(answer=message, uncertainty=message,
                                      meta={"answer_version": n, "change_type": "refine", "reason": reason,
                                            "retrieval_required": True, "ttft_ms": state.get("ttft_ms")}),
        }

    # -- entry -----------------------------------------------------------------------

    async def run(
        self, utterance: str, turn_id: str, state: Dict[str, Any], totals: Dict[str, float], gold: List[str],
        sub_gold_map: Optional[Dict[str, List[str]]] = None, is_unans: Optional[bool] = None,
    ) -> Dict[str, Any]:
        e = self.engine
        ledger = e.ledger
        inputs = planner_inputs(ledger)
        plan = await self.planner.plan(utterance, inputs["sub_intents"], inputs["claims"])
        self.last_plan = plan
        charge(e.clock, e.latency.plan_s)
        self._charge_tokens(totals, plan.tokens_in, plan.tokens_out)
        e._publish(PlanCompletedEvent(turn_id=turn_id, timestamp=e._now(), payload=plan.payload()))

        checkpoint = ledger.checkpoint()
        try:
            if plan.relation == "modifies":
                return await self._modifies(utterance, plan, turn_id, state, totals, gold)
            return await self._adds(utterance, plan, turn_id, state, totals, gold, sub_gold_map, is_unans)
        except Exception as exc:
            ledger.restore(checkpoint)
            if isinstance(exc, PreservationError) and self.cfg.strict_preservation:
                raise
            logger.exception("refinement failed; previous ledger version kept")
            return self._no_change(
                turn_id, utterance, plan, state, totals, 0,
                [{"sub_intent_id": sid, "reason": "refinement_error", "text": f"could not apply '{utterance}' to the previous answer"}
                 for sid in (plan.affected_sub_intents or ["session"])],
                reason="refinement_error", error=repr(exc),
            )

    # -- modifies ----------------------------------------------------------------------

    async def _modifies(
        self, utterance: str, plan: DeltaPlan, turn_id: str, state: Dict[str, Any], totals: Dict[str, float], gold: List[str],
    ) -> Dict[str, Any]:
        e = self.engine
        ledger = e.ledger
        affected = plan.affected_sub_intents
        sub_text = {s.id: s.text for s in ledger.sub_intents}
        ledger.add_constraint(Constraint(
            constraint_id=f"k{len(ledger.constraints) + 1}", text=utterance, turn_id=turn_id, affects=list(affected),
        ))

        # Delta retrieval: <= len(delta_queries) <= 2 retrievals; sub-intents share a query round-robin.
        groups: Dict[int, List[str]] = {}
        for i, sid in enumerate(affected):
            groups.setdefault(i % len(plan.delta_queries), []).append(sid)
        evidence: Dict[str, List[Any]] = {}
        core_query: Dict[str, str] = {}
        fresh = 0
        for qi, sids in sorted(groups.items()):
            query = form_delta_query(plan.delta_queries[qi], sub_text.get(sids[0], ""))
            rid = f"{turn_id}:delta{qi + 1}"
            hit = e.cache.lookup(query, turn_id=turn_id) if e.cache is not None else None
            if hit is not None:
                hit.entry.used = True
                chunks, source = list(hit.entry.evidence), "cache"
            else:
                chunks, source = list(e.retrieval_engine.retrieve(query)), "fresh"
                fresh += 1
                if e.cache is not None:
                    e.cache.store(rid, query, chunks, turn_id=turn_id)
            e._publish(SpeculativeRetrievalEvent(
                turn_id=turn_id, retrieval_id=rid, query=query, trigger="refinement", is_early=False,
                source=source, chunks_count=len(chunks), timestamp=e._now(),
                latency_ms=(e.latency.retrieval_s if source == "fresh" else e.latency.cache_hit_s) * 1000.0 if e.clock else 0.0,
            ))
            ids = [chunk_id_of(c, str(i)) for i, c in enumerate(chunks)]
            for sid in sids:
                evidence[sid] = chunks
                core_query[sid] = plan.delta_queries[qi]
                e.remember_evidence(sid, chunks[: self.cfg.evidence_per_sub_intent])
                e._publish(SubIntentRetrievalEvent(
                    turn_id=turn_id, sub_intent_id=sid, retrieval_query=query, source=source, retrieved_chunk_ids=ids,
                    evidence_chunk_ids=ids[: self.cfg.evidence_per_sub_intent], expected_chunk_ids=gold, timestamp=e._now(),
                    evidence=evidence_details(chunks[: self.cfg.evidence_per_sub_intent]),
                ))
        charge(e.clock, e.latency.retrieval_s if fresh else e.latency.cache_hit_s)

        # Sufficiency gate on the delta evidence, per affected sub-intent (gated on what is new).
        sufficient: List[str] = []
        uncertainty: List[Dict[str, Any]] = []
        for sid in affected:
            outcome = normalize_gate_result(e.sufficiency.evaluate(core_query[sid], evidence[sid]))
            ok = outcome.sufficient
            e._publish(SubIntentCompletionEvent(
                turn_id=turn_id, sub_intent_id=sid, status="completed" if ok else "suppressed", is_suppressed=not ok,
                is_uncertain=not ok, reason="refinement_sufficient" if ok else "refinement_insufficient", timestamp=e._now(),
                **gate_fields(outcome, e.sufficiency),
            ))
            if ok:
                sufficient.append(sid)
            else:
                uncertainty.append({
                    "sub_intent_id": sid, "reason": "refinement_insufficient",
                    "gate_scores": {} if outcome.score is None else {"score": outcome.score},
                    "text": f"could not verify how {utterance.strip().rstrip('.')} affects {sub_text.get(sid, sid).rstrip('?')}",
                })
        if not sufficient:
            return self._no_change(turn_id, utterance, plan, state, totals, fresh, uncertainty, reason="refinement_insufficient",
                                   insufficient_sub_intents=list(affected))

        # Rewriter: only the claims of the affected (and sufficiently evidenced) sub-intents.
        affected_claims = ledger.claims_for(sufficient)
        held = {sid: list(e.session_evidence.get(sid, {})) for sid in sufficient}
        rw = await self.rewriter.rewrite(utterance, sufficient, affected_claims, {sid: evidence[sid] for sid in sufficient}, held)
        self.last_rewrite = rw
        charge(e.clock, e.latency.draft_s)
        self._charge_tokens(totals, rw.tokens_in, rw.tokens_out)
        if rw.fallback:
            return self._no_change(
                turn_id, utterance, plan, state, totals, fresh,
                uncertainty + [{"sub_intent_id": sid, "reason": "refinement_rewrite_failed",
                                "text": f"could not revise the answer for {sub_text.get(sid, sid).rstrip('?')}"} for sid in sufficient],
                reason=f"rewrite_fallback:{rw.fallback_reason}", rewrite_input_claim_ids=rw.input_claim_ids,
            )

        # Verify revised and new claims only, against delta evidence + evidence held for these sub-intents.
        pool: Dict[str, Any] = {}
        for sid in sufficient:
            for ch in evidence[sid][: self.cfg.evidence_per_sub_intent]:
                pool[chunk_id_of(ch, "")] = ch
            pool.update(e.session_evidence.get(sid, {}))
        changing = {d.claim_id for d in rw.decisions if d.action in ("revise", "retract")}
        preserved = [c for c in ledger.get_verified_claims() if c.claim_id not in changing]
        originals = {c.claim_id: c for c in affected_claims}
        candidates: List[Claim] = [
            originals[d.claim_id].model_copy(update={"text": d.text, "doc_ids": list(d.cites or [])})
            for d in rw.decisions if d.action == "revise"
        ] + [Claim(claim_id=f"new_{i}", text=n.text, doc_ids=list(n.cites), sub_intent_id=n.sub_intent_id)
             for i, n in enumerate(rw.new_claims)]
        charge(e.clock, e.latency.verify_s)
        passed, groundedness = self._verify(candidates, list(pool.values()), preserved, turn_id)

        # Contradiction guard against preserved claims (unaffected + kept).
        flags = guard([(c.claim_id, c.text) for c in candidates if c.claim_id in passed], preserved)
        contradictions = [{"claim_id": k, **v} for k, v in flags.items()]
        revision_rejected, new_rejected = [], []
        decisions: List[Decision] = []
        for d in rw.decisions:
            if d.action == "revise" and (d.claim_id not in passed or d.claim_id in flags):
                decisions.append(Decision(d.claim_id, "keep"))  # revert to the previous text
                if d.claim_id not in passed:
                    revision_rejected.append(d.claim_id)
                else:
                    uncertainty.append({"sub_intent_id": originals[d.claim_id].sub_intent_id, "reason": "contradiction",
                                        "text": f"unverified revision: {d.text}", "contradiction": flags[d.claim_id]})
            else:
                decisions.append(Decision(d.claim_id, d.action, d.text, d.cites))
        new_claims: List[NewClaim] = []
        for i, n in enumerate(rw.new_claims):
            key = f"new_{i}"
            if key not in passed:
                new_rejected.append(n.text)
            elif key in flags:
                uncertainty.append({"sub_intent_id": n.sub_intent_id, "reason": "contradiction",
                                    "text": f"unverified addition: {n.text}", "contradiction": flags[key]})
            else:
                new_claims.append(NewClaim(n.sub_intent_id, n.text, list(n.cites)))
        for cid in revision_rejected:
            logger.info("revision_rejected: %s", cid)

        result = apply_delta(
            ledger, sufficient, decisions, new_claims, turn_id, rewrite_input_claim_ids=rw.input_claim_ids,
            uncertainty=uncertainty, strict=self.cfg.strict_preservation,
        )

        # Emit: answer_chunk only for added and revised claims (first_token on the first), then the diff.
        changed = set(result.added) | set(result.revised)
        emitted: List[Claim] = [c for c in ledger.claims_in_sub_intent_order() if c.claim_id in changed]
        for i, c in enumerate(emitted):
            if i == 0:
                e._first_token(turn_id, c.text, state)
            e._publish(AnswerChunk(turn_id=turn_id, answer_version=result.to_version, claim_id=c.claim_id, text=c.text,
                                   cites=list(c.doc_ids), first_token=i == 0, ts_s=e._now()))
        emitted_text = " ".join(f"{c.text} [{', '.join(c.doc_ids)}]" for c in emitted)
        e._count_cites(emitted_text, list(pool.values()), totals)
        rendered = " ".join(f"{c.text} [{', '.join(c.doc_ids)}]" for c in ledger.claims_in_sub_intent_order())
        e._publish(AnswerDelta(turn_id=turn_id, text_delta=emitted_text, is_final=True, ops=result.ops,
                               change_type="refine", rendered_answer=rendered, answer_version=result.to_version,
                               hashes={c.claim_id: c.text_hash for c in ledger.get_verified_claims()}))
        self._transition(turn_id, result.transition_payload(
            applied=True, change_type="refine", relation=plan.relation, contradictions=contradictions,
            revision_rejected=revision_rejected, new_claim_rejected=new_rejected,
            insufficient_sub_intents=[sid for sid in affected if sid not in sufficient],
            post_refinement_conflicts=conflicts_in(ledger.get_verified_claims()),
        ))
        e._complete(turn_id, rendered, state, totals, fresh)
        return {
            "output": rendered, "refined": True, "relation": plan.relation, "delta": result,
            "retrieval_calls": fresh, "groundedness": groundedness if groundedness is not None else 0.0,
            "latency_ms": (e._now() - state["t_start"]) * 1000.0, "resolved_count": len(sufficient), "suppressed_count": 0,
            "chunks": list(pool.values()),
            "turn_output": TurnOutput(
                sub_queries=[form_delta_query(plan.delta_queries[0], sub_text.get(s, "")) for s in affected],
                answer=rendered, citations=sorted({c for cl in ledger.get_verified_claims() for c in cl.doc_ids}),
                uncertainty=" ".join(u["text"] for u in uncertainty if u.get("text")) or None,
                meta={"answer_version": result.to_version, "change_type": "refine", "retrieval_required": True,
                      "ttft_ms": state.get("ttft_ms"), "tokens_in": int(totals["tokens_in"]), "tokens_out": int(totals["tokens_out"])},
            ),
        }

    # -- adds / unrelated ----------------------------------------------------------------

    async def _adds(
        self, utterance: str, plan: DeltaPlan, turn_id: str, state: Dict[str, Any], totals: Dict[str, float],
        gold: List[str], sub_gold_map: Optional[Dict[str, List[str]]], is_unans: Optional[bool],
    ) -> Dict[str, Any]:
        """Phase 5 flow for the new content only; new sub-intents get fresh ids; existing claims untouched."""
        e = self.engine
        ledger = e.ledger
        before = ledger.get_verified_claims()
        pre_hashes = {c.claim_id: c.text_hash for c in before}
        n0 = ledger.answer_version
        offset = len(ledger.sub_intents)
        if e.config.enable_multi_intent and e.decomposer is not None:
            result = await e._multi_intent_path(utterance, turn_id, gold, sub_gold_map, is_unans, state, totals, 0, sub_id_offset=offset)
        else:
            result = await e._single_path(utterance, turn_id, gold, sub_gold_map, is_unans, state, totals, 0, sub_id=f"sub_{offset + 1}")

        after = {c.claim_id: c for c in ledger.get_verified_claims()}
        identical = sum(1 for cid, h in pre_hashes.items() if cid in after and compute_text_hash(after[cid].text, after[cid].doc_ids) == h)
        ok = identical == len(pre_hashes)
        if not ok:
            if self.cfg.strict_preservation:
                raise PreservationError("an existing claim changed on an adds turn")
            logger.error("preservation assertion failed on an adds turn")
        added = [cid for cid in after if cid not in pre_hashes]
        ops = [{"op": "keep", "claim_id": cid} for cid in pre_hashes] + [
            {"op": "add", "claim_id": cid, "text": after[cid].text, "cites": list(after[cid].doc_ids)} for cid in added
        ]
        e._publish(AnswerDelta(turn_id=turn_id, text_delta=result.get("output", ""), is_final=True, ops=ops, change_type="add",
                               rendered_answer=ledger.render(), answer_version=ledger.answer_version,
                               hashes={c.claim_id: c.text_hash for c in ledger.get_verified_claims()}))
        self._transition(turn_id, {
            "from": n0, "to": ledger.answer_version, "kept": list(pre_hashes), "revised": [], "retracted": [], "added": added,
            "unchanged_hashes_ok": ok, "rewrite_input_claim_ids": [], "unaffected": list(pre_hashes),
            "preserved_checked": len(pre_hashes), "preserved_identical": identical, "applied": ledger.answer_version > n0,
            "change_type": "add", "relation": plan.relation, "fallback": plan.fallback,
            "post_refinement_conflicts": conflicts_in(ledger.get_verified_claims()),
        })
        result.update({"refined": False, "relation": plan.relation})
        return result
