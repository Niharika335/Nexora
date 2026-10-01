"""Batch Turn Engine coordinating retrieval, sufficiency gate, claim drafting, verification, and streaming."""

import asyncio
import contextlib
import logging
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple

from slrag.config import AppConfig, DEFAULT_CONFIG, SlragConfig
from slrag.contracts.events import (
    AnswerChunk,
    AnswerDelta,
    AnswerVersionTransitionEvent,
    ControllerDecisionEvent,
    ReconcileCompletedEvent,
    SpeculationOutcomeEvent,
    VerificationEvent,
    BaseEvent,
    Claim,
    ClaimStatus,
    FirstTokenEmissionEvent,
    Ledger,
    MultiIntentResolutionEvent,
    ReconciliationEvent,
    RetrievalEvent,
    RetrievalItem,
    SpeculativeRetrievalEvent,
    SubIntent,
    SubIntentCompletionEvent,
    SubIntentRetrievalEvent,
    SuppressionEvent,
    TurnCompleteEvent,
    TurnOutput,
    TurnStartEvent,
    TurnSummary,
    UtteranceFinalEvent,
)
from slrag.answer.predraft import DraftResult, PendingDraft, Predrafter, VerifiedDraft
from slrag.answer.reconcile import ReconcileDecision, ReconcileOutcome, decide, kept_claims
from slrag.control.controller import RetrievalController
from slrag.control.t2_llm import T2Classifier
from slrag.eval.clock import DEFAULT_LATENCY, LatencyModel, charge
from slrag.gateway.session import SessionContext
from slrag.nlp.lemmas import CorpusVocab
from slrag.pipeline.dag_planner import DAGPlanner
from slrag.pipeline.drafter import ClaimDrafter
from slrag.pipeline.intent_decomposer import IntentDecomposer
from slrag.pipeline.ledger import ClaimLedger
from slrag.pipeline.llm import LLMServiceWrapper
from slrag.pipeline.presenter import Presenter
from slrag.pipeline.reconciler import PlanReconciler
from slrag.pipeline.streaming import StreamingTurn
from slrag.pipeline.subintent_executor import SubIntentExecutor
from slrag.pipeline.sufficiency import SufficiencyGate
from slrag.pipeline.subintent_executor import evidence_details
from slrag.pipeline.suppression import SuppressionController, gate_fields, normalize_gate_result
from slrag.pipeline.verifier import FailClosedVerifier
from slrag.plan.planner import QueryPlanner
from slrag.retrieval.cache import EvidenceCache, chunk_id_of
from slrag.retrieval.engine import HybridRetrievalEngine, SearchResult
from slrag.retrieval.instrumented import instrumented_search
from slrag.retrieval.quota import apply_quota
from slrag.telemetry.bus import GLOBAL_BUS, SessionStampedBus, TelemetryBus
from slrag.telemetry.metrics import MetricsCollector, GLOBAL_METRICS

logger = logging.getLogger(__name__)


class BatchTurnEngine:
    """End-to-end turn orchestrator executing retrieval, sufficiency, generation, verification, and streaming."""

    def __init__(
        self,
        config: AppConfig = DEFAULT_CONFIG,
        engine: Optional[HybridRetrievalEngine] = None,
        bus: Optional[TelemetryBus] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        self.config = config
        self.engine = engine or HybridRetrievalEngine(config)
        self.bus = bus or GLOBAL_BUS
        self.metrics = metrics or GLOBAL_METRICS

        self.llm_service = LLMServiceWrapper(config.llm)
        self.sufficiency_gate = SufficiencyGate(config.sufficiency)
        self.verifier = FailClosedVerifier(config.verifier)

    def _get_or_init_session_ledger(self, session_ctx: SessionContext) -> ClaimLedger:
        """Retrieve or initialize the session-scoped claim ledger."""
        if "ledger" not in session_ctx.custom_state:
            session_ctx.custom_state["ledger"] = ClaimLedger(session_ctx.session_id)
        return session_ctx.custom_state["ledger"]

    async def process_turn_stream(
        self,
        session_ctx: SessionContext,
        utterance: str,
        turn_id: str,
        prefetched: Optional[List[SearchResult]] = None,
        pending: Optional[PendingDraft] = None,
        early_calls: int = 0,
    ) -> AsyncIterator[BaseEvent]:
        """Execute a complete interaction turn and yield output stream events.

        `pending` is the commit-stage pre-draft of this live turn (Phase 8): it is reconciled against
        the final evidence (stands / refined / redone) before drafting.

        `prefetched` is evidence retrieved early by the streaming controller (reused under the
        cache rule); when given, no new search runs and the retrieval event is logged with mode "cache".
        """
        t_start = time.perf_counter()
        session_id = session_ctx.session_id
        ledger = self._get_or_init_session_ledger(session_ctx)
        seq = 100

        # 1. Hybrid Retrieval with instrumentation (or reuse of early-retrieved evidence)
        if prefetched is not None:
            retrieval_results = list(prefetched)
            await self.bus.emit(RetrievalEvent(
                session_id=session_id, turn_id=turn_id, seq=seq, query=utterance, mode="cache",
                top_k=len(retrieval_results), latency_ms=0.0,
                results=[
                    RetrievalItem(chunk_id=r.chunk_id, score=float(r.score), rank=r.rank, source_scores=r.source_scores,
                                  section_title=r.section_title, doc_id=r.doc_id)
                    for r in retrieval_results
                ],
            ))
            self.metrics.record_event("retrieval")
        else:
            retrieval_results = await instrumented_search(
                engine=self.engine,
                query=utterance,
                turn_id=turn_id,
                session_id=session_id,
                mode="hybrid",
                top_k=5,
                seq=seq,
                bus=self.bus,
                metrics=self.metrics,
            )
        seq += 1

        # 2. Sufficiency Gate
        passed_sufficiency, dense_score, coverage_score, reason = await self.sufficiency_gate.evaluate_and_emit(
            query=utterance,
            results=retrieval_results,
            turn_id=turn_id,
            session_id=session_id,
            seq=seq,
            bus=self.bus,
        )
        seq += 1

        sub_intent_id = f"{turn_id}:q1"
        gate_scores = {"dense_top1": round(dense_score, 4), "coverage": round(coverage_score, 4)}
        ledger.add_sub_intent(SubIntent(
            id=sub_intent_id, text=utterance,
            status="answerable" if passed_sufficiency else "insufficient", sufficiency=gate_scores,
        ))

        self._session_evidence(session_ctx).setdefault(sub_intent_id, {}).update(
            {r.chunk_id: r for r in retrieval_results[:4]}
        )

        if not passed_sufficiency:
            if pending is not None:
                self.discard_live_pending(session_ctx, pending, turn_id, "insufficient_evidence")
            ledger.add_uncertainty(sub_intent_id, "insufficient_evidence", gate_scores)
            # Fallback for insufficient knowledge
            fallback_msg = "I do not have enough context in the knowledge base to answer this question accurately."
            delta_evt = AnswerDelta(
                session_id=session_id,
                turn_id=turn_id,
                seq=seq,
                text_delta=fallback_msg,
                is_final=True,
            )
            seq += 1
            await self.bus.emit(delta_evt)
            yield delta_evt

            latency_ms = (time.perf_counter() - t_start) * 1000.0
            turn_summary = TurnSummary(
                session_id=session_id,
                turn_id=turn_id,
                seq=seq,
                utterance=utterance,
                claims_count=0,
                verified_count=0,
                rejected_count=0,
                tokens_in=self.llm_service.estimate_tokens(utterance),
                tokens_out=self.llm_service.estimate_tokens(fallback_msg),
                cost=0.0,
                latency_ms=round(latency_ms, 2),
                status="insufficient_context",
                ttft_ms=round(latency_ms, 2),
                retrieval_calls=early_calls + (0 if prefetched is not None else 1),
            )
            await self.bus.emit(turn_summary)  # logged here, like every other event this stream yields
            yield turn_summary
            self.metrics.record_turn()
            self.metrics.record_latency(latency_ms, kind="turn")
            return

        # 3. LLM Generation & Claim Drafting
        retrieved_chunks_map = {r.chunk_id: self.engine.chunks_map[r.chunk_id] for r in retrieval_results if r.chunk_id in self.engine.chunks_map}
        allowed_doc_ids: Set[str] = set(retrieved_chunks_map.keys())

        # Phase 8: reconcile a commit-stage pre-draft against the final evidence.
        reuse: List[Claim] = []
        draft_results = list(retrieval_results)
        speculation: Dict[str, Any] = {}
        tokens_in = tokens_out = 0
        cost = 0.0
        if pending is not None:
            rec = await self._reconcile_live(session_ctx, pending, utterance, retrieval_results, turn_id)
            speculation = rec["summary"]
            tokens_in, tokens_out, cost = pending.tokens_in, pending.tokens_out, pending.cost
            if rec["decision"].reuses_predraft:
                reuse = [c.model_copy(update={"sub_intent_id": sub_intent_id, "turn_id": turn_id}) for c in rec["kept"]]
                draft_results = [r for r in retrieval_results if r.chunk_id in rec["decision"].new_chunk_ids]

        drafted_claims: List[Claim] = []
        llm_calls = pending.llm_calls if pending is not None else 0
        if draft_results:  # no pre-draft / redone: every chunk; refined: only the new chunks; stands: nothing
            llm_calls += 1
            chunks_payload = [
                {"chunk_id": r.chunk_id, "text": r.text, "section_title": r.section_title}
                for r in draft_results
            ]
            llm_result = await self.llm_service.generate_grounded_response(
                query=utterance,
                retrieved_chunks=chunks_payload,
            )
            tokens_in, tokens_out, cost = tokens_in + llm_result.tokens_in, tokens_out + llm_result.tokens_out, cost + llm_result.cost
            drafter = ClaimDrafter(allowed_doc_ids=allowed_doc_ids)
            drafted_claims = [
                c.model_copy(update={"sub_intent_id": sub_intent_id})
                for c in drafter.draft_from_raw_claims(llm_result.raw_claims, turn_id=turn_id)
            ]

        # 4. Fail-Closed Verification on every newly drafted claim (reused pre-draft claims were verified
        # against the same, still-present evidence at commit time)
        verified_claims: List[Claim] = list(reuse)
        rejected_claims: List[Claim] = []
        previously_verified = ledger.get_verified_claims()

        for claim in drafted_claims:
            updated_claim, passed = await self.verifier.verify_and_emit(
                claim=claim,
                retrieved_chunks_map=retrieved_chunks_map,
                allowed_doc_ids=allowed_doc_ids,
                previously_verified=previously_verified,
                turn_id=turn_id,
                session_id=session_id,
                seq=seq,
                bus=self.bus,
            )
            seq += 1
            if passed:
                verified_claims.append(updated_claim)
            else:
                rejected_claims.append(updated_claim)

        # 5. Update Claim Ledger with verified claims
        if verified_claims:
            version_before = ledger.answer_version
            before_ids = [c.claim_id for c in ledger.get_verified_claims()]
            ledger.add_verified_claims(verified_claims, turn_id=turn_id)
            for i, c in enumerate(verified_claims):
                self.bus.publish(AnswerChunk(
                    session_id=session_id, turn_id=turn_id, answer_version=ledger.answer_version, claim_id=c.claim_id,
                    text=c.text, cites=list(c.doc_ids), first_token=i == 0,
                ))
            self._record_live_transition(session_ctx, turn_id, version_before, [c.claim_id for c in verified_claims], before_ids)
        else:
            ledger.add_uncertainty(sub_intent_id, "no_verified_claims", gate_scores)
        
        ledger_snapshot = await ledger.emit_snapshot(turn_id=turn_id, seq=seq, bus=self.bus)
        seq += 1

        # 6. Stream AnswerDelta containing ONLY verified claims
        if verified_claims:
            answer_parts = [f"{c.text} [{', '.join(c.doc_ids)}]" for c in verified_claims]
            verified_answer_text = " ".join(answer_parts)
        else:
            verified_answer_text = "I could not verify any facts to answer your query with confidence."

        # Emit in chunks to simulate streaming deltas
        ttft_ms: Optional[float] = None
        words = verified_answer_text.split()
        chunk_size = max(1, len(words) // 3)
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            delta_str = " ".join(chunk_words)
            if i > 0:
                delta_str = " " + delta_str
            is_final = (i + chunk_size >= len(words))

            delta_evt = AnswerDelta(
                session_id=session_id,
                turn_id=turn_id,
                seq=seq,
                text_delta=delta_str,
                is_final=is_final,
            )
            seq += 1
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t_start) * 1000.0
            await self.bus.emit(delta_evt)
            yield delta_evt
            await asyncio.sleep(0.01)  # Micro-yield for stream simulation

        # Yield ledger update
        yield ledger_snapshot

        # 7. Emit Turn Summary
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        turn_summary = TurnSummary(
            session_id=session_id,
            turn_id=turn_id,
            seq=seq,
            utterance=utterance,
            claims_count=len(drafted_claims) + len(reuse),
            verified_count=len(verified_claims),
            rejected_count=len(rejected_claims),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=round(cost, 6),
            latency_ms=round(latency_ms, 2),
            status="completed",
            ttft_ms=None if ttft_ms is None else round(ttft_ms, 2),
            retrieval_calls=early_calls + (0 if prefetched is not None else 1),
            llm_calls=llm_calls,
            llm_mode=self.llm_service.last_mode,
            **speculation,
        )
        await self.bus.emit(turn_summary)
        self.metrics.record_turn()
        self.metrics.record_latency(latency_ms, kind="turn")

        yield turn_summary


    # -- Phase 7: late-detail refinement on live turns ---------------------------------

    @staticmethod
    def _session_evidence(session_ctx: SessionContext) -> Dict[str, Dict[str, Any]]:
        return session_ctx.custom_state.setdefault("session_evidence", {})

    @staticmethod
    def _transitions(session_ctx: SessionContext) -> List[Dict[str, Any]]:
        return session_ctx.custom_state.setdefault("transitions", [])

    def _record_live_transition(self, session_ctx: SessionContext, turn_id: str, version_before: int, added: List[str],
                                kept: Optional[List[str]] = None) -> None:
        ledger = self._get_or_init_session_ledger(session_ctx)
        if ledger.answer_version == version_before:
            return
        transitions = self._transitions(session_ctx)
        last_to = transitions[-1]["to"] if transitions else 0
        live = {c.claim_id: c for c in ledger.get_verified_claims()}
        self.bus.publish(AnswerDelta(
            session_id=session_ctx.session_id, turn_id=turn_id, text_delta="", is_final=True,
            ops=[{"op": "keep", "claim_id": cid} for cid in kept or []]
            + [{"op": "add", "claim_id": cid, "text": live[cid].text, "cites": list(live[cid].doc_ids)} for cid in added if cid in live],
            change_type="initial" if version_before == 0 else "add", rendered_answer=ledger.render(),
            answer_version=ledger.answer_version, hashes={cid: c.text_hash for cid, c in live.items()},
        ))
        payload = {
            "from": version_before, "to": ledger.answer_version, "kept": list(kept or []), "revised": [], "retracted": [], "added": added,
            "unchanged_hashes_ok": True, "rewrite_input_claim_ids": [], "applied": True,
            "change_type": "initial" if version_before == 0 else "add",
            "lineage_ok": version_before == last_to and ledger.answer_version == version_before + 1,
        }
        transitions.append(payload)
        self.bus.publish(AnswerVersionTransitionEvent(session_id=session_ctx.session_id, turn_id=turn_id, payload=payload))

    # -- Phase 8: commit-stage pre-draft on live turns -------------------------------

    def _live_predrafter(self, session_ctx: SessionContext) -> Predrafter:
        ledger = self._get_or_init_session_ledger(session_ctx)

        def draft(text: str, chunks: List[Any]) -> DraftResult:
            payload = [{"chunk_id": c.chunk_id, "text": c.text, "section_title": getattr(c, "section_title", "")} for c in chunks]
            res = self.llm_service.generate_grounded_response_sync(text, payload)
            return DraftResult(res.text, res.tokens_in, res.tokens_out, res.cost, raw=res.raw_claims)

        def verify(result: DraftResult, chunks: List[Any]) -> VerifiedDraft:
            chunks_map = {c.chunk_id: self.engine.chunks_map[c.chunk_id] for c in chunks if c.chunk_id in self.engine.chunks_map}
            claims = ClaimDrafter(set(chunks_map)).draft_from_raw_claims(result.raw or [], turn_id="")
            previous = ledger.get_verified_claims()
            passed = [c.model_copy(update={"verification_status": ClaimStatus.VERIFIED}) for c in claims
                      if self.verifier.verify_claim(c, chunks_map, set(chunks_map), previous)[0]]
            return VerifiedDraft(" ".join(f"{c.text} [{', '.join(c.doc_ids)}]" for c in passed), passed, len(passed), len(claims))

        def gate(text: str, chunks: List[Any]):
            passed, _, coverage, _ = self.sufficiency_gate.evaluate(text, chunks) if chunks else (False, 0.0, 0.0, "")
            return passed, coverage

        return Predrafter(draft, verify, gate, bus=SessionStampedBus(self.bus, session_ctx.session_id),
                          concurrency=self.config.speculation.draft_concurrency)

    async def start_live_predraft(self, stream: StreamingTurn) -> Optional[PendingDraft]:
        """COMMIT on /ws/stream: start the background pre-draft on the commit-time evidence."""
        session_ctx = stream.session_ctx
        text = stream.state.buffer

        async def evidence(sub_id: str, query: str) -> List[Any]:
            chunks = await stream.commit_evidence(query)
            if chunks is None and stream.cache is not None:
                chunks = stream.cache.peek(query)
            if chunks is None:
                chunks = await asyncio.to_thread(self.engine.search, query, "hybrid", 5)
            return list(chunks)

        return self._live_predrafter(session_ctx).start(stream.turn_id, text, [("q1", text)], evidence, epoch=stream.state.epoch)

    def discard_live_pending(self, session_ctx: SessionContext, pending: PendingDraft, turn_id: str, reason: str) -> None:
        """The live turn took another path (presentation, refinement, insufficient evidence): pre-draft wasted."""
        pending.cancel(reason)
        self.bus.publish(ReconcileCompletedEvent(
            session_id=session_ctx.session_id, turn_id=turn_id, outcome=ReconcileOutcome.REDONE.value, reason=reason,
            wasted_tokens=pending.tokens, predraft_tokens=pending.tokens, discarded_predrafts=len(pending.sub_drafts),
        ))

    async def _reconcile_live(
        self, session_ctx: SessionContext, pending: PendingDraft, utterance: str, results: List[SearchResult], turn_id: str,
    ) -> Dict[str, Any]:
        timeout = self.config.speculation.predraft_timeout_s
        before_end = pending.done
        evidence_ok = await pending.wait_evidence(timeout)
        decision = decide(utterance, [("q1", utterance)], {"q1": list(results)}, pending,
                          reuse_cos=self.config.cache.reuse_cos, max_new_chunks=self.config.speculation.refine_max_new_chunks,
                          ready=evidence_ok)
        late_ms, kept, discarded = 0.0, [], 0
        if decision.reuses_predraft:
            t0 = time.perf_counter()
            if await pending.wait(timeout):
                late_ms = 0.0 if before_end else (time.perf_counter() - t0) * 1000.0
                sub = decision.subs["q1"].pending
                kept, discarded = kept_claims(sub, decision.subs["q1"].dropped_ids) if sub is not None else ([], 0)
            else:
                decision = ReconcileDecision(ReconcileOutcome.REDONE, pending.cancel_reason or "timeout")
        if not decision.reuses_predraft:
            pending.cancel(pending.cancel_reason or "reconciled_redone")
        wasted = 0 if decision.reuses_predraft else pending.tokens
        new_claims_expected = len(decision.new_chunk_ids) if decision.outcome == ReconcileOutcome.REFINED else 0
        self.bus.publish(ControllerDecisionEvent(
            session_id=session_ctx.session_id, turn_id=turn_id, decision="COMMIT_RECONCILE", tier="T1", reason=decision.outcome.value,
            payload={"detail": decision.reason, "new_chunk_ids": decision.new_chunk_ids, "dropped_chunk_ids": decision.dropped_chunk_ids},
        ))
        self.bus.publish(ReconcileCompletedEvent(
            session_id=session_ctx.session_id, turn_id=turn_id, outcome=decision.outcome.value, reason=decision.reason,
            stands=len(kept), refined=new_claims_expected, redone=0 if decision.reuses_predraft else len(results),
            wasted_tokens=wasted, predraft_tokens=pending.tokens, new_chunk_ids=decision.new_chunk_ids,
            dropped_chunk_ids=decision.dropped_chunk_ids, discarded_claims=discarded,
            discarded_predrafts=0 if decision.reuses_predraft else len(pending.sub_drafts),
            predraft_ready_before_end=before_end, predraft_late_ms=round(late_ms, 3), timed_out=pending.cancel_reason == "timeout",
        ))
        summary = {"reconcile_outcome": decision.outcome.value, "predraft_ready_before_end": before_end, "wasted_tokens": wasted,
                   "discarded_predrafts": 0 if decision.reuses_predraft else len(pending.sub_drafts)}
        return {"decision": decision, "kept": kept, "summary": summary}

    def should_refine(self, session_ctx: SessionContext) -> bool:
        """A live content turn is refined (not answered from scratch) once the session has an answer."""
        ledger = session_ctx.custom_state.get("ledger")
        return bool(self.config.refinement.enabled and ledger is not None and ledger.answer_version >= 1)

    def _refinement_engine(self, session_ctx: SessionContext) -> "TurnEngine":
        """Per-session TurnEngine + RefinementRouter sharing this session's ledger, evidence cache,
        held evidence and version lineage, so live and replay refinement run the same code."""
        state = session_ctx.custom_state
        if "refinement_engine" not in state:
            from slrag.replay.baselines import DrafterAdapter, GateAdapter, VerifierAdapter  # local: baselines imports pipeline

            ledger = self._get_or_init_session_ledger(session_ctx)
            bus = _LiveRefinementBus(self.bus, session_ctx.session_id)
            cfg = SlragConfig(enable_cascade=False, enable_multi_intent=True, enable_refinement=True, enable_verifier=True,
                              restart_on_late_detail=False, top_k=5, app=self.config)
            engine = TurnEngine(
                config=cfg, retrieval_engine=_LiveRetriever(self.engine), llm=None,
                verifier=VerifierAdapter(self.config, bus, ledger), sufficiency=GateAdapter(self.config), ledger=ledger,
                drafter=DrafterAdapter(self.config), bus=bus,
                vocab=CorpusVocab(self.engine.bm25_index.idf_table) if self.engine.chunks_map else None,
            )
            engine.cache = state.get("evidence_cache")  # the live Phase 4 cache: delta retrieval looks here first
            engine.session_evidence = self._session_evidence(session_ctx)
            engine.transitions = self._transitions(session_ctx)
            state["refinement_engine"] = engine
        return state["refinement_engine"]

    async def refine_turn_stream(self, session_ctx: SessionContext, utterance: str, turn_id: str) -> AsyncIterator[BaseEvent]:
        """Later live content turn: delta planner -> modifies (delta retrieval, rewrite, verify, apply)
        or adds (Phase 5 flow for the new content). Internal events go straight to telemetry; the
        client-facing ones (answer_chunk, answer_delta, turn_summary) are yielded to be sent and logged."""
        t_start = time.perf_counter()
        engine = self._refinement_engine(session_ctx)
        bus: _LiveRefinementBus = engine.bus
        result = await engine.refine_utterance(utterance, turn_id)
        for evt in bus.drain_client_events():
            yield evt
        delta = result.get("delta")
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        status = "refined" if result.get("refined") else ("added" if result.get("relation") in ("adds", "unrelated") else "refinement_no_change")
        yield TurnSummary(
            session_id=session_ctx.session_id, turn_id=turn_id, seq=200, utterance=utterance,
            claims_count=len(delta.revised) + len(delta.added) if delta else 0,
            verified_count=len(delta.revised) + len(delta.added) if delta else 0, rejected_count=0,
            tokens_in=int(engine.last_totals.get("tokens_in", 0)), tokens_out=int(engine.last_totals.get("tokens_out", 0)),
            cost=float(engine.last_totals.get("cost", 0.0)), latency_ms=round(latency_ms, 2), status=status,
            ttft_ms=result.get("turn_output").meta.get("ttft_ms") if result.get("turn_output") is not None else None,
            retrieval_calls=int(result.get("retrieval_calls", 0) or 0), llm_calls=int(engine.last_totals.get("llm_calls", 0)),
        )
        self.metrics.record_turn()
        self.metrics.record_latency(latency_ms, kind="turn")

    async def present_turn_stream(self, session_ctx: SessionContext, instruction: str, turn_id: str) -> AsyncIterator[BaseEvent]:
        """Presentation-only turn (controller SUPPRESS): re-render the session ledger with zero retrievals."""
        t_start = time.perf_counter()
        ledger = self._get_or_init_session_ledger(session_ctx)
        result = await Presenter().render(instruction, ledger)
        session_id = session_ctx.session_id
        yield SuppressionEvent(session_id=session_id, turn_id=turn_id, reason="presentation_restructure",
                               fallback=result.fallback, bullets=len(result.bullets))
        yield AnswerDelta(session_id=session_id, turn_id=turn_id, seq=100, text_delta=result.rendered, is_final=True)
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        yield TurnSummary(
            session_id=session_id, turn_id=turn_id, seq=101, utterance=instruction, claims_count=0, verified_count=0,
            rejected_count=0, tokens_in=0, tokens_out=0, cost=0.0, latency_ms=round(latency_ms, 2), status="presentation",
        )
        self.metrics.record_turn()
        self.metrics.record_latency(latency_ms, kind="turn")


class _LiveRetriever:
    """retrieve(query) over the live index, at BatchTurnEngine's depth (top 5, hybrid)."""

    def __init__(self, engine: HybridRetrievalEngine, top_k: int = 5):
        self.engine = engine
        self.top_k = top_k

    def retrieve(self, query: str) -> List[SearchResult]:
        return self.engine.search(query, mode="hybrid", top_k=self.top_k)


class _LiveRefinementBus:
    """Bus for the live refinement engine: internal events go straight to telemetry; client-facing
    events are held so /ws/stream can send them and log them once (as it does for other turns)."""

    CLIENT_EVENTS = {"answer_chunk", "answer_delta", "turn_summary"}

    def __init__(self, telemetry: Any, session_id: str):
        self.telemetry = telemetry
        self.session_id = session_id
        self._client: List[BaseEvent] = []

    def publish(self, event: BaseEvent) -> None:
        if event.session_id != self.session_id:
            event = event.model_copy(update={"session_id": self.session_id})
        if event.event_type in self.CLIENT_EVENTS:
            self._client.append(event)
        else:
            self.telemetry.publish(event)

    def drain_client_events(self) -> List[BaseEvent]:
        out, self._client = self._client, []
        return out


def _cited(claim: Claim) -> str:
    """ "Claim text [cid]." -- citation before the final punctuation (same format as the verifier output)."""
    body = claim.text.rstrip()
    end = body[-1] if body[-1:] in (".", "!", "?") else "."
    return f"{body.rstrip('.!?')} [{', '.join(claim.doc_ids)}]{end}"


def extract_cites_claims(text: str) -> List[Claim]:
    return ClaimDrafter(set()).draft_from_text(text, turn_id="") if text else []


CITE_RE = re.compile(r"\[([^\[\]]+)\]")


def extract_cites(text: str) -> List[str]:
    """Ordered, de-duplicated cite IDs from bracketed citations like "[A§1, B§2]"."""
    cites: List[str] = []
    for group in CITE_RE.findall(text):
        for part in (p.strip() for p in group.split(",")):
            if part and part not in cites:
                cites.append(part)
    return cites


def gold_sub_intent_count(sub_gold_map: Optional[Dict[str, List[str]]]) -> Optional[int]:
    """Gold sub-intents that the corpus can answer (unanswerable ones have no gold chunks)."""
    if not sub_gold_map:
        return None
    return sum(1 for ids in sub_gold_map.values() if ids)


class TurnEngine:
    """Streaming turn orchestrator used by the replay harness for ours / B0 / B1.

    Each turn is streamed in as word chunks. With `enable_cascade` every chunk goes through
    the RetrievalController (early PROVISIONAL / COMMIT retrieval into the evidence cache,
    or SUPPRESS -> Presenter). At utterance_end the multi-intent path (planner -> parallel
    cache-aware retrieval with quotas -> per-sub-intent gate / draft / verify) or the single
    query path (baselines) runs. Components are duck-typed: retrieval_engine.retrieve(query),
    sufficiency.evaluate(query, chunks), drafter.draft(...), verifier.verify(...).
    """

    def __init__(
        self,
        config: SlragConfig,
        retrieval_engine: Any,
        llm: Any,
        verifier: Any,
        sufficiency: Any,
        ledger: Any,
        drafter: Any,
        bus: Any,
        clock: Optional[Any] = None,
        vocab: Optional[CorpusVocab] = None,
        planner_llm: Optional[Any] = None,
        presenter: Optional[Presenter] = None,
        latency: LatencyModel = DEFAULT_LATENCY,
        delta_llm: Optional[Any] = None,
        rewrite_llm: Optional[Any] = None,
    ):
        self.config = config
        self.retrieval_engine = retrieval_engine
        self.llm = llm
        self.verifier = verifier
        self.sufficiency = sufficiency
        self.ledger = ledger
        self.drafter = drafter
        self.bus = bus
        self.clock = clock
        self.latency = latency
        self.presenter = presenter or Presenter()
        app = config.app

        self.cache: Optional[EvidenceCache] = None
        self.controller: Optional[RetrievalController] = None
        if config.enable_cascade:
            self.cache = EvidenceCache(
                ttl_ms=config.speculative_cache_ttl_ms, max_size=config.speculative_cache_max_size,
                bus=bus, clock=clock, reuse_cos=app.cache.reuse_cos,
            )
            self.controller = RetrievalController(app.controller, vocab=vocab, t2=T2Classifier(timeout_s=app.controller.t2_timeout_s))

        self.planner = QueryPlanner(app.planner, llm_plan_fn=planner_llm)
        if config.enable_multi_intent:
            self.decomposer: Optional[IntentDecomposer] = IntentDecomposer(bus=bus, clock=clock, planner=self.planner)
            self.dag_planner: Optional[DAGPlanner] = DAGPlanner()
            self.sub_executor: Optional[SubIntentExecutor] = SubIntentExecutor(
                retrieval_engine=retrieval_engine, cache=self.cache, bus=bus, clock=clock,
                per_sub_quota=app.planner.quota_per_subquery, global_quota=app.planner.quota_global, latency=latency,
            )
            self.suppression: Optional[SuppressionController] = SuppressionController(
                sufficiency_gate=sufficiency, bus=bus, clock=clock,
                uncertain_band=(app.sufficiency.uncertain_low, app.sufficiency.uncertain_high),
            )
            self.reconciler: Optional[PlanReconciler] = PlanReconciler(bus=bus, clock=clock)
        else:
            self.decomposer = self.dag_planner = self.sub_executor = self.suppression = self.reconciler = None

        # Phase 8: commit-stage pre-drafting (speculation.mode == full and speculation.predraft).
        self.speculation = app.speculation
        self.predrafter: Optional[Predrafter] = None
        if config.enable_cascade and config.enable_multi_intent and self.speculation.predraft_on:
            self.predrafter = Predrafter(
                self._predraft_draft, self._predraft_verify, self._predraft_gate, bus=bus, clock=clock, latency=latency,
                concurrency=self.speculation.draft_concurrency,
            )

        # Phase 7: session evidence store (sub-intent -> {chunk_id: chunk}) and version lineage.
        self.session_evidence: Dict[str, Dict[str, Any]] = {}
        self.transitions: List[Dict[str, Any]] = []
        self.last_totals: Dict[str, float] = {}
        self._chunked_turns: Set[str] = set()
        self.refinement: Optional[Any] = None
        if config.enable_refinement and app.refinement.enabled:
            from slrag.engine.turn_engine import RefinementRouter  # local import: engine imports pipeline modules
            from slrag.llm import get_llm

            backend = get_llm(app.llm)  # llm.backend: heuristic (default) | ollama
            self.refinement = RefinementRouter(
                self, app.refinement, delta_llm=delta_llm or backend.delta_fn, rewrite_llm=rewrite_llm or backend.rewrite_fn,
            )

    # -- helpers ---------------------------------------------------------------

    def _now(self) -> float:
        return self.clock.time() if self.clock else time.time()

    def _publish(self, event: Any) -> None:
        self.bus.publish(event)

    def _stream_chunks(self, text: str) -> List[str]:
        words = text.split()
        n = max(1, self.latency.words_per_chunk)
        return [" ".join(words[i:i + n]) for i in range(0, len(words), n)] or [text]

    def _usage(self) -> Dict[str, float]:
        usage = getattr(self.drafter, "last_usage", None) or {}
        return {k: float(usage.get(k, 0.0)) for k in ("tokens_in", "tokens_out", "cost")}

    def _draft(self, query: str, chunks: List[Any], totals: Dict[str, float]) -> str:
        # The drafter keeps per-call usage state; share the pre-draft lock so a still-running pre-draft
        # thread (e.g. after a timeout) cannot interleave with this call.
        with (self.predrafter.draft_lock if self.predrafter is not None else contextlib.nullcontext()):
            output = self.drafter.draft(query=query, chunks=chunks, temperature=self.config.temperature, seed=self.config.seed)
            usage = self._usage()
        for k, v in usage.items():
            totals[k] += v
        totals["llm_calls"] = totals.get("llm_calls", 0) + 1
        return output

    def _verify(self, draft: str, chunks: List[Any], turn_id: str) -> Tuple[str, Optional[float]]:
        """Fail-closed: only the verifier's surviving text is emitted. Returns (text, groundedness)."""
        if not (self.config.enable_verifier and self.verifier):
            return draft, None
        res = self.verifier.verify(draft=draft, chunks=chunks, turn_id=turn_id)
        return getattr(res, "verified_text", draft), float(getattr(res, "groundedness_score", 1.0))

    def _first_token(self, turn_id: str, text: str, state: Dict[str, Any]) -> None:
        if not state["first_emitted"]:
            self._publish(FirstTokenEmissionEvent(turn_id=turn_id, token=text.split()[0] if text else "", timestamp=self._now()))
            state["first_emitted"] = True
            state["ttft_ms"] = (self._now() - state["t_end"]) * 1000.0

    @staticmethod
    def _count_cites(text: str, chunks: List[Any], totals: Dict[str, float]) -> None:
        """Audit emitted citations against the evidence the answer was drafted from."""
        evidence = {chunk_id_of(c, "") for c in chunks}
        cites = extract_cites(text)
        totals["emitted_cites"] = totals.get("emitted_cites", 0) + len(cites)
        totals["hallucinated_cites"] = totals.get("hallucinated_cites", 0) + sum(1 for c in cites if c not in evidence)

    def _emit_claims(self, turn_id: str, claim_ids: List[str]) -> None:
        """answer_chunk per recorded claim, in emission order (telemetry for the trace UI's answer panel)."""
        if self.ledger is None or not hasattr(self.ledger, "get_claim"):
            return
        for cid in claim_ids:
            claim = self.ledger.get_claim(cid)
            if claim is None:
                continue
            first = turn_id not in self._chunked_turns
            self._chunked_turns.add(turn_id)
            self._publish(AnswerChunk(
                turn_id=turn_id, answer_version=getattr(self.ledger, "answer_version", 0) + 1, claim_id=cid,
                text=claim.text, cites=list(claim.doc_ids), first_token=first, ts_s=self._now(),
            ))

    def _record_claims(self, sub_id: str, text: str, turn_id: str, verified: bool) -> List[str]:
        """Write each atomic claim ("sentence [cites].") of an answer into the ledger via the reconciler."""
        claim_ids: List[str] = []
        for claim in ClaimDrafter(set()).draft_from_text(text, turn_id=turn_id) or [Claim(text=text)]:
            if self.reconciler is not None:
                claim_ids.append(self.reconciler.reconcile(
                    self.ledger, sub_intent_id=sub_id, claim_text=claim.text, is_update=False, turn_id=turn_id,
                    doc_ids=claim.doc_ids, verified=verified,
                ))
            elif self.ledger is not None and hasattr(self.ledger, "add_or_update_claim"):
                claim_ids.append(self.ledger.add_or_update_claim(
                    sub_id, claim.text, doc_ids=claim.doc_ids, turn_id=turn_id, verified=verified,
                ))
        self._emit_claims(turn_id, claim_ids)
        return claim_ids

    # -- Phase 8 pre-draft hooks ------------------------------------------------------

    def _predraft_draft(self, text: str, chunks: List[Any]) -> DraftResult:
        output = self.drafter.draft(query=text, chunks=chunks, temperature=self.config.temperature, seed=self.config.seed)
        usage = self._usage()
        return DraftResult(output, int(usage["tokens_in"]), int(usage["tokens_out"]), usage["cost"])

    def _predraft_verify(self, draft: DraftResult, chunks: List[Any]) -> VerifiedDraft:
        if self.config.enable_verifier and self.verifier:
            try:
                res = self.verifier.verify(draft=draft.text, chunks=chunks, turn_id="", publish=False)
            except TypeError:  # duck-typed verifiers without a publish flag
                res = self.verifier.verify(draft=draft.text, chunks=chunks, turn_id="")
            text = getattr(res, "verified_text", draft.text)
            supported = len(getattr(res, "supported", []))
            total = supported + len(getattr(res, "unsupported", []))
        else:
            text = draft.text
            supported = total = len(ClaimDrafter(set()).draft_from_text(text, turn_id=""))
        claims = [c.model_copy(update={"verification_status": ClaimStatus.VERIFIED})
                  for c in ClaimDrafter(set()).draft_from_text(text, turn_id="")] if text else []
        return VerifiedDraft(text, claims, supported, total)

    def _predraft_gate(self, text: str, chunks: List[Any]) -> Tuple[bool, Optional[float]]:
        if not chunks:
            return False, 0.0
        outcome = normalize_gate_result(self.sufficiency.evaluate(text, chunks))
        return outcome.sufficient, outcome.score

    async def _on_commit(self, stream: StreamingTurn) -> Optional[PendingDraft]:
        """COMMIT: start the background pre-draft on the commit-time evidence (never blocks the chunk path)."""
        assert self.predrafter is not None
        plan = stream.committed_plan or await self.planner.plan(stream.state.buffer)
        subs = [(f"sub_{i + 1}", sq.text) for i, sq in enumerate(plan.sub_queries)]
        holder: Dict[str, PendingDraft] = {}
        turn_id = stream.turn_id

        async def evidence(sub_id: str, text: str) -> List[Any]:
            chunks = await stream.commit_evidence(text)
            if chunks is None and self.cache is not None:
                chunks = self.cache.peek(text)
            if chunks is None:  # no commit retrieval serves this sub-query: an early retrieval of its own
                chunks = list(await asyncio.to_thread(self.retrieval_engine.retrieve, text))
                holder["p"].retrieval_calls += 1
                self._publish(SpeculativeRetrievalEvent(
                    turn_id=turn_id, retrieval_id=f"{turn_id}:{sub_id}:predraft", query=text, trigger="predraft", is_early=True,
                    chunks_count=len(chunks), timestamp=self._now(),
                    latency_ms=self.latency.retrieval_s * 1000.0 if self.clock else 0.0,
                ))
                if self.cache is not None:
                    self.cache.store(f"{turn_id}:{sub_id}:predraft", text, chunks, turn_id=turn_id)
            quota = self.config.app.planner.quota_per_subquery
            return apply_quota({sub_id: chunks}, per_sub=quota, global_cap=self.config.app.planner.quota_global)[sub_id]

        holder["p"] = self.predrafter.start(
            turn_id, stream.state.buffer, subs, evidence, epoch=stream.state.epoch, evidence_ready_at=stream.commit_ready_at(),
        )
        return holder["p"]

    async def _await_predraft(self, pending: PendingDraft, turn_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """utterance_end: wait for the pre-draft (bounded by speculation.predraft_timeout_s). Late drafts are
        waited for (predraft_late); a timed-out one is cancelled and the turn falls through to the normal path."""
        timeout = self.speculation.predraft_timeout_s
        info: Dict[str, Any] = {"ready": False, "late_ms": 0.0, "before_end": False, "timed_out": False}
        if self.clock is not None and pending.ready_at is not None:
            late_s = max(0.0, pending.ready_at - state["t_end"])
            info["before_end"] = pending.ready_at <= state["t_end"]
            if late_s > timeout:
                pending.cancel("timeout")
                info.update(timed_out=True, late_ms=timeout * 1000.0)
                charge(self.clock, timeout)
                return info
            info["ready"] = await pending.wait(timeout)
            if info["ready"]:
                self.clock.advance_to(pending.ready_at)
                info["late_ms"] = late_s * 1000.0
        else:
            info["before_end"] = pending.done
            t0 = time.perf_counter()
            info["ready"] = await pending.wait(timeout)
            info["late_ms"] = 0.0 if info["before_end"] else (time.perf_counter() - t0) * 1000.0
        info["timed_out"] = info["timed_out"] or pending.cancel_reason == "timeout"
        if info["ready"] and info["late_ms"] > 0:
            self._publish(SpeculationOutcomeEvent(
                turn_id=turn_id, outcome="predraft_late", reason=f"waited {info['late_ms']:.0f} ms", timestamp=self._now(),
            ))
        return info

    def _discard_pending(self, pending: PendingDraft, turn_id: str, reason: str) -> None:
        """A pre-draft whose turn took another path (presentation): cancelled, all its tokens wasted."""
        pending.cancel(reason)
        self._publish(ReconcileCompletedEvent(
            turn_id=turn_id, outcome=ReconcileOutcome.REDONE.value, reason=reason, wasted_tokens=pending.tokens,
            predraft_tokens=pending.tokens, discarded_predrafts=len(pending.sub_drafts), timestamp=self._now(),
        ))

    def remember_evidence(self, sub_id: str, chunks: List[Any]) -> None:
        """Merge evidence into the session evidence store for a sub-intent."""
        store = self.session_evidence.setdefault(sub_id, {})
        for i, c in enumerate(chunks):
            store.setdefault(chunk_id_of(c, str(i)), c)

    def _register_sub_intent(self, sub_id: str, text: str, answerable: bool) -> None:
        if self.ledger is not None and hasattr(self.ledger, "add_sub_intent"):
            self.ledger.add_sub_intent(SubIntent(id=sub_id, text=text, status="answerable" if answerable else "insufficient"))

    def record_transition(self, turn_id: str, payload: Dict[str, Any]) -> None:
        """Publish answer_version_transition; lineage_ok = this step continues the previous one."""
        last_to = self.transitions[-1]["to"] if self.transitions else 0
        payload = dict(payload)
        payload["lineage_ok"] = payload["from"] == last_to and payload["to"] in (payload["from"], payload["from"] + 1)
        self.transitions.append(payload)
        self._publish(AnswerVersionTransitionEvent(turn_id=turn_id, payload=payload, timestamp=self._now()))

    def _record_uncertainty(self, sub_id: str, reason: str, score: Optional[float] = None) -> None:
        if self.ledger is not None and hasattr(self.ledger, "add_uncertainty"):
            self.ledger.add_uncertainty(sub_id, reason, {} if score is None else {"score": score})

    def _complete(self, turn_id: str, output: str, state: Dict[str, Any], totals: Dict[str, float], retrieval_calls: int,
                  speculation: Optional[Dict[str, Any]] = None) -> None:
        self._publish(TurnCompleteEvent(
            **(speculation or {}),
            turn_id=turn_id,
            ttft_ms=state.get("ttft_ms") or 0.0,
            total_latency_ms=(self._now() - state["t_start"]) * 1000.0,
            output_text=output,
            retrieval_calls=retrieval_calls,
            tokens_in=int(totals["tokens_in"]),
            tokens_out=int(totals["tokens_out"]),
            est_cost_usd=totals["cost"],
            emitted_cites=int(totals.get("emitted_cites", 0)),
            hallucinated_cites=int(totals.get("hallucinated_cites", 0)),
            llm_calls=int(totals.get("llm_calls", 0)),
            llm_mode=getattr(self.drafter, "llm_mode", None) or self.config.app.llm.backend,
            timestamp=self._now(),
        ))

    # -- turn ------------------------------------------------------------------

    async def execute_turn(
        self,
        query: str,
        turn_id: str = "",
        expected_chunk_ids: Optional[List[str]] = None,
        is_unanswerable_ground_truth: Optional[bool] = None,
        sub_gold_map: Optional[Dict[str, List[str]]] = None,
        late_detail: Optional[str] = None,
        stream_chunks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        state: Dict[str, Any] = {"t_start": self._now(), "first_emitted": False, "ttft_ms": None}
        totals = {"tokens_in": 0.0, "tokens_out": 0.0, "cost": 0.0, "emitted_cites": 0.0, "hallucinated_cites": 0.0}
        self._publish(TurnStartEvent(turn_id=turn_id, query=query, timestamp=state["t_start"]))
        gold = list(expected_chunk_ids or [])
        for ids in (sub_gold_map or {}).values():
            gold.extend(i for i in ids if i not in gold)

        if self.config.restart_on_late_detail and late_detail:
            return await self._restart_on_late_detail(query, late_detail, turn_id, state, totals)

        final_query = f"{query} {late_detail}" if late_detail else query

        # Phase 7: a later content turn of a session is planned as a delta at utterance_end, so the
        # controller still runs (T0 presentation first) but does not dispatch early raw-utterance retrievals.
        refine = self.refinement is not None and self.refinement.should_route()

        # 1. Stream the utterance through the controller (early retrieval happens here).
        stream: Optional[StreamingTurn] = None
        if self.controller is not None and self.cache is not None:
            stream = StreamingTurn(
                turn_id, self.controller, self.retrieval_engine.retrieve, self.cache, bus=self.bus, clock=self.clock,
                ledger=self.ledger, planner=self.planner if self.config.enable_multi_intent else None, latency=self.latency,
                # speculation.mode == off (A3 arm): the controller still decides (T0 presentation) but dispatches nothing.
                defer_dispatch=refine or not self.speculation.provisional_on,
                on_commit=self._on_commit if (self.predrafter is not None and not refine) else None,
            )
        for i, chunk in enumerate(stream_chunks or self._stream_chunks(final_query)):
            if i > 0:
                charge(self.clock, self.latency.chunk_interval_s)
            if stream is not None:
                await stream.on_chunk(chunk)

        # 2. utterance_end
        charge(self.clock, self.latency.utterance_end_gap_s)
        state["t_end"] = self._now()
        self._publish(UtteranceFinalEvent(turn_id=turn_id, query=final_query, timestamp=state["t_end"]))
        early_calls = 0
        pending: Optional[PendingDraft] = None
        if stream is not None:
            await stream.settle()
            early_calls = stream.fresh_retrieval_count
            pending = stream.pending
            if stream.suppressed:
                if pending is not None:
                    self._discard_pending(pending, turn_id, "presentation_turn")
                    stream.state.predraft_status = "wasted"
                stream.close(retrieval_required=False)
                return await self._present(final_query, turn_id, state, totals)

        if refine:
            result = await self.refinement.run(final_query, turn_id, state, totals, gold, sub_gold_map, is_unanswerable_ground_truth)
            if stream is not None:
                stream.close(retrieval_required=True)
            return result

        version_before = getattr(self.ledger, "answer_version", None)
        claims_before = [c.claim_id for c in self.ledger.get_verified_claims()] if version_before is not None else []
        if self.config.enable_multi_intent and self.decomposer is not None:
            result = await self._multi_intent_path(final_query, turn_id, gold, sub_gold_map, is_unanswerable_ground_truth, state, totals,
                                                   early_calls, pending=pending)
            if pending is not None and stream is not None:
                stream.state.predraft_claims = pending.claims
                stream.state.predraft_status = "committed" if result.get("reconcile_outcome") in ("stands", "refined") else "wasted"
                result["predraft_status"] = stream.state.predraft_status
        else:
            result = await self._single_path(final_query, turn_id, gold, sub_gold_map, is_unanswerable_ground_truth, state, totals, early_calls)
        if version_before is not None and self.ledger.answer_version != version_before:
            added = [c.claim_id for c in self.ledger.get_verified_claims() if c.claim_id not in set(claims_before)]
            live = {c.claim_id: c for c in self.ledger.get_verified_claims()}
            self._publish(AnswerDelta(
                turn_id=turn_id, text_delta=result.get("output", ""), is_final=True,
                ops=[{"op": "keep", "claim_id": cid} for cid in claims_before]
                + [{"op": "add", "claim_id": cid, "text": live[cid].text, "cites": list(live[cid].doc_ids)} for cid in added],
                change_type="initial" if version_before == 0 else "add", rendered_answer=self.ledger.render(),
                answer_version=self.ledger.answer_version, hashes={cid: c.text_hash for cid, c in live.items()},
            ))
            self.record_transition(turn_id, {
                "from": version_before, "to": self.ledger.answer_version, "kept": claims_before, "revised": [], "retracted": [],
                "added": added, "unchanged_hashes_ok": True, "rewrite_input_claim_ids": [], "applied": True,
                "change_type": "initial" if version_before == 0 else "add",
            })

        if stream is not None:
            stream.close(retrieval_required=True)
            result["retrieval_events"] = [
                {"retrieval_id": d.retrieval_id, "query": d.query, "trigger": d.trigger, "source": d.source, "stale": d.stale}
                for d in stream.dispatches
            ] + result.get("retrieval_events", [])
        return result

    async def refine_utterance(self, utterance: str, turn_id: str) -> Dict[str, Any]:
        """Live /ws/stream entry for a later content turn: the controller already ran on the live
        chunks (early retrieval paused), so go straight to the refinement path at utterance_end."""
        assert self.refinement is not None, "refinement is disabled for this engine"
        state: Dict[str, Any] = {"t_start": self._now(), "first_emitted": False, "ttft_ms": None}
        totals = {"tokens_in": 0.0, "tokens_out": 0.0, "cost": 0.0, "emitted_cites": 0.0, "hallucinated_cites": 0.0}
        self.last_totals = totals
        self._publish(TurnStartEvent(turn_id=turn_id, query=utterance, timestamp=state["t_start"]))
        state["t_end"] = self._now()
        self._publish(UtteranceFinalEvent(turn_id=turn_id, query=utterance, timestamp=state["t_end"]))
        return await self.refinement.run(utterance, turn_id, state, totals, [])

    async def _restart_on_late_detail(self, query: str, late_detail: str, turn_id: str, state: Dict[str, Any], totals: Dict[str, float]) -> Dict[str, Any]:
        """Baseline behaviour: a late detail discards the first pass and restarts retrieval + drafting."""
        combined_query = f"{query} {late_detail}"
        state["t_end"] = self._now()
        self._publish(UtteranceFinalEvent(turn_id=turn_id, query=combined_query, timestamp=state["t_end"]))
        self.retrieval_engine.retrieve(query)
        charge(self.clock, self.latency.retrieval_s)
        self._publish(ReconciliationEvent(turn_id=turn_id, reconciliation_type="restart", affected_claim_ids=[], timestamp=self._now()))
        restarted_chunks = self.retrieval_engine.retrieve(combined_query)
        charge(self.clock, self.latency.retrieval_s + self.latency.draft_s)
        output = self._draft(combined_query, restarted_chunks, totals)
        output, _ = self._verify(output, restarted_chunks, turn_id)
        self._first_token(turn_id, output, state)
        self._complete(turn_id, output, state, totals, retrieval_calls=2)
        return {"output": output, "restarted": True, "latency_ms": (self._now() - state["t_start"]) * 1000.0, "retrieval_calls": 2}

    async def _present(self, instruction: str, turn_id: str, state: Dict[str, Any], totals: Dict[str, float]) -> Dict[str, Any]:
        """SUPPRESS: re-render the ledger with the Presenter. No retrieval, no new claims, same answer_version."""
        result = await self.presenter.render(instruction, self.ledger)
        charge(self.clock, self.latency.present_s)
        self._publish(SuppressionEvent(
            turn_id=turn_id, reason="presentation_restructure", fallback=result.fallback, bullets=len(result.bullets), timestamp=self._now(),
        ))
        output = result.rendered
        self._first_token(turn_id, output, state)
        # The re-rendered answer on the stream (as the live gateway does for presentation turns).
        self._publish(AnswerDelta(turn_id=turn_id, text_delta=output, is_final=True, answer_version=result.answer_version))
        self._complete(turn_id, output, state, totals, retrieval_calls=0)
        turn_output = TurnOutput(
            retrieval_events=[], sub_queries=[], answer=output,
            citations=sorted({c for b in result.bullets for c in b["cites"]}),
            meta={"answer_version": result.answer_version, "retrieval_required": False, "reason": "presentation_restructure",
                  "presenter_fallback": result.fallback, "ttft_ms": state["ttft_ms"]},
        )
        return {"output": output, "suppressed": True, "retrieval_calls": 0, "resolved_count": 0, "suppressed_count": 0,
                "turn_output": turn_output, "latency_ms": (self._now() - state["t_start"]) * 1000.0}

    async def _single_path(
        self, query: str, turn_id: str, gold: List[str], sub_gold_map: Optional[Dict[str, List[str]]],
        is_unans: Optional[bool], state: Dict[str, Any], totals: Dict[str, float], early_calls: int,
        sub_id: str = "sub_1",
    ) -> Dict[str, Any]:
        """One retrieval for the whole utterance at utterance_end (cache-aware), then gate -> draft -> verify."""
        hit = self.cache.lookup(query, turn_id=turn_id) if self.cache else None
        fresh_calls = 0
        if hit is not None:
            hit.entry.used = True
            chunks = hit.entry.evidence
            if self.clock is not None:
                self.clock.advance_to(max(self._now() + self.latency.cache_hit_s, hit.entry.ready_at))
        else:
            chunks = self.retrieval_engine.retrieve(query)
            fresh_calls = 1
            charge(self.clock, self.latency.retrieval_s)
            self._publish(SpeculativeRetrievalEvent(
                turn_id=turn_id, retrieval_id=f"{turn_id}:final", query=query, trigger="final", is_early=False,
                chunks_count=len(chunks), timestamp=self._now() - (self.latency.retrieval_s if self.clock else 0.0),
                latency_ms=self.latency.retrieval_s * 1000.0 if self.clock else 0.0,
            ))
            if self.cache:
                self.cache.store(f"{turn_id}:final", query, list(chunks), turn_id=turn_id)
        self._publish(SubIntentRetrievalEvent(
            turn_id=turn_id, sub_intent_id=sub_id, retrieval_query=query, source="cache" if hit else "fresh",
            retrieved_chunk_ids=[chunk_id_of(c, str(i)) for i, c in enumerate(chunks)],
            evidence_chunk_ids=[chunk_id_of(c, str(i)) for i, c in enumerate(chunks)],
            expected_chunk_ids=gold, timestamp=self._now(), evidence=evidence_details(chunks),
        ))

        outcome = normalize_gate_result(self.sufficiency.evaluate(query, chunks))
        gate_info = gate_fields(outcome, self.sufficiency)
        self._register_sub_intent(sub_id, query, outcome.sufficient)
        self.remember_evidence(sub_id, list(chunks)[:4])
        groundedness: Optional[float] = None
        if not outcome.sufficient:
            output = f"Cannot answer: {outcome.reason or 'lacks evidence'}"
            self._record_uncertainty(sub_id, "insufficient_evidence", outcome.score)
            self._publish(SubIntentCompletionEvent(
                turn_id=turn_id, sub_intent_id=sub_id, status="suppressed", is_suppressed=True, is_uncertain=True,
                is_unanswerable_ground_truth=is_unans, reason="insufficient_evidence", timestamp=self._now(), **gate_info,
            ))
            self._first_token(turn_id, output, state)
        else:
            self._publish(SubIntentCompletionEvent(
                turn_id=turn_id, sub_intent_id=sub_id, status="completed", is_unanswerable_ground_truth=is_unans, timestamp=self._now(),
                **gate_info,
            ))
            charge(self.clock, self.latency.draft_s)
            draft = self._draft(query, chunks, totals)
            charge(self.clock, self.latency.verify_s)
            output, groundedness = self._verify(draft, chunks, turn_id)
            if output:
                self._first_token(turn_id, output, state)
                self._count_cites(output, chunks, totals)
                self._record_claims(sub_id, output, turn_id, verified=groundedness is not None)
                if self.ledger is not None and hasattr(self.ledger, "bump_answer_version"):
                    self.ledger.bump_answer_version()
            else:
                self._record_uncertainty(sub_id, "no_verified_claims")

        resolved = 1 if outcome.sufficient and output else 0
        self._publish(MultiIntentResolutionEvent(
            turn_id=turn_id, total_sub_intents=1, resolved_count=resolved, suppressed_count=0 if outcome.sufficient else 1,
            resolution_recall=float(resolved), gold_sub_intents=gold_sub_intent_count(sub_gold_map), timestamp=self._now(),
        ))
        retrieval_calls = early_calls + fresh_calls
        self._complete(turn_id, output, state, totals, retrieval_calls)
        return {"output": output, "latency_ms": (self._now() - state["t_start"]) * 1000.0,
                "groundedness": groundedness if groundedness is not None else 0.0, "chunks": chunks,
                "retrieval_calls": retrieval_calls, "resolved_count": resolved,
                "suppressed_count": 0 if outcome.sufficient else 1}

    async def _retrieve_wave(self, wave: List[Dict[str, Any]], stage_idx: int, turn_id: str, gold: List[str]) -> Dict[str, List[Any]]:
        t_wave = self._now()
        evidence = await self.sub_executor.execute_wave(wave, stage_idx=stage_idx, turn_id=turn_id, expected_chunk_ids=gold)
        for sub in wave:
            if self.sub_executor.last_sources.get(sub["sub_intent_id"]) == "fresh":
                self._publish(SpeculativeRetrievalEvent(
                    turn_id=turn_id, retrieval_id=f"{turn_id}:{sub['sub_intent_id']}:final", query=sub["text"],
                    trigger="final", is_early=False, timestamp=t_wave, source="fresh",
                    latency_ms=self.latency.retrieval_s * 1000.0 if self.clock else 0.0,
                ))
        return evidence

    async def _multi_intent_path(
        self, query: str, turn_id: str, gold: List[str], sub_gold_map: Optional[Dict[str, List[str]]],
        is_unans: Optional[bool], state: Dict[str, Any], totals: Dict[str, float], early_calls: int,
        sub_id_offset: int = 0, pending: Optional[PendingDraft] = None,
    ) -> Dict[str, Any]:
        """Planner -> DAG waves -> parallel retrieval with quotas -> per-sub-intent gate / draft / verify -> ledger.

        `sub_id_offset` numbers new sub-intents after the session's existing ones (Phase 7 adds turns).
        With a commit-stage `pending` pre-draft (Phase 8) the final evidence of every wave is gathered
        first, the pre-draft is reconciled against it (stands | refined | redone), and each sub-intent
        reuses its verified pre-draft claims (drafting only new chunks) or is drafted from scratch."""
        assert self.decomposer and self.dag_planner and self.sub_executor and self.suppression and self.reconciler
        wait: Optional[Dict[str, Any]] = None
        replay_ok: Optional[bool] = None
        if pending is not None and self.clock is not None:
            # Replay determinism: let the pre-draft computation finish (wall clock) before the final
            # retrieval, so cache contents and token accounting never depend on thread timing. This does
            # not move the virtual clock: virtual time is charged only when the drafts are reused.
            replay_ok = await pending.wait(self.speculation.predraft_timeout_s)
        sub_intents = await self.decomposer.adecompose(query, turn_id=turn_id, id_offset=sub_id_offset)
        charge(self.clock, self.latency.plan_s)
        calls_before = self.sub_executor.retrieval_calls

        resolved = suppressed = uncertain = 0
        answers: List[str] = []
        uncertainty_lines: List[str] = []
        all_chunks: List[Any] = []
        groundedness_scores: List[float] = []
        drafted_any = False

        waves = self.dag_planner.plan(sub_intents)
        decision = None
        prefetched: Dict[str, List[Any]] = {}
        counts = {"stands": 0, "refined": 0, "redone": 0, "discarded_claims": 0, "discarded_predrafts": 0, "wasted_tokens": 0}
        if pending is not None:
            for stage_idx, wave in enumerate(waves):
                prefetched.update(await self._retrieve_wave(wave, stage_idx, turn_id, gold))
            # Decide on the pre-draft's evidence first (it lands long before its drafts); only a pre-draft
            # that will be reused is waited for, so a discarded one never delays the answer.
            evidence_ok = replay_ok if replay_ok is not None else await pending.wait_evidence(self.speculation.predraft_timeout_s)
            if evidence_ok and self.clock is not None and pending.evidence_ready_at is not None:
                self.clock.advance_to(pending.evidence_ready_at)
            decision = decide(
                query, [(s["sub_intent_id"], s["text"]) for s in sub_intents], prefetched, pending,
                reuse_cos=self.config.app.cache.reuse_cos, max_new_chunks=self.speculation.refine_max_new_chunks,
                ready=evidence_ok,
            )
            if decision.reuses_predraft:
                wait = await self._await_predraft(pending, turn_id, state)
                if not wait["ready"]:
                    decision = ReconcileDecision(ReconcileOutcome.REDONE, pending.error and "predraft_error" or pending.cancel_reason or "timeout")
            else:
                before_end = pending.ready_at <= state["t_end"] if pending.ready_at is not None else pending.done
                timed_out = pending.cancel_reason == "timeout"
                pending.cancel(pending.cancel_reason or "reconciled_redone")
                wait = {"ready": False, "late_ms": 0.0, "before_end": before_end, "timed_out": timed_out}
            self._publish(ControllerDecisionEvent(
                turn_id=turn_id, decision="COMMIT_RECONCILE", tier="T1", reason=decision.outcome.value, timestamp=self._now(),
                payload={"detail": decision.reason, "new_chunk_ids": decision.new_chunk_ids, "dropped_chunk_ids": decision.dropped_chunk_ids},
            ))
            charge(self.clock, self.latency.reconcile_s)
            # Pre-draft tokens were spent whatever the outcome; they are wasted unless its claims are reused.
            totals["tokens_in"] += pending.tokens_in
            totals["tokens_out"] += pending.tokens_out
            totals["cost"] += pending.cost
            totals["llm_calls"] = totals.get("llm_calls", 0) + pending.llm_calls
            discarded = list(pending.sub_drafts.values()) if not decision.reuses_predraft else decision.unused_predrafts
            counts["wasted_tokens"] = sum(sd.tokens for sd in discarded)
            counts["discarded_predrafts"] = sum(1 for sd in discarded if sd.draft is not None)

        for stage_idx, wave in enumerate(waves):
            evidence = prefetched if pending is not None else await self._retrieve_wave(wave, stage_idx, turn_id, gold)

            for sub in wave:
                s_id, s_text = sub["sub_intent_id"], sub["text"]
                chunks = evidence.get(s_id, [])
                all_chunks.extend(chunks)
                decision_sub = decision.subs.get(s_id) if decision is not None and decision.reuses_predraft else None
                gate = self.suppression.evaluate(s_id, s_text, chunks, turn_id=turn_id, is_unanswerable_ground_truth=is_unans)
                self._register_sub_intent(s_id, s_text, not gate["suppressed"])
                self.remember_evidence(s_id, chunks)
                if gate["suppressed"]:
                    suppressed += 1
                    if decision_sub is not None and decision_sub.pending is not None and decision_sub.pending.draft is not None:
                        counts["wasted_tokens"] += decision_sub.pending.tokens
                        counts["discarded_predrafts"] += 1
                    self._record_uncertainty(s_id, "insufficient_evidence", gate["score"])
                    uncertainty_lines.append(f"{s_text} could not be verified from the retrieved corpus.")
                    continue

                pre = decision_sub.pending if decision_sub is not None else None
                if pre is not None and pre.verified is not None:
                    text, groundedness = self._reconciled_sub(decision_sub, s_text, chunks, turn_id, state, totals, counts)
                    drafted_any = True
                else:
                    # Drafts run in parallel: the first emitted sub-intent pays the draft latency.
                    charge(self.clock, (0.0 if drafted_any else self.latency.draft_s) + self.latency.verify_s)
                    drafted_any = True
                    draft = self._draft(s_text, chunks, totals)
                    text, groundedness = self._verify(draft, chunks, turn_id)
                    if decision is not None:
                        counts["redone"] += len(extract_cites_claims(text))
                if groundedness is not None:
                    groundedness_scores.append(groundedness)
                if not text:
                    self._record_uncertainty(s_id, "no_verified_claims")
                    uncertainty_lines.append(f"{s_text} could not be verified from the retrieved corpus.")
                    continue

                if gate["uncertain"]:
                    uncertain += 1
                self._first_token(turn_id, text, state)
                self._count_cites(text, chunks, totals)
                self._record_claims(s_id, text, turn_id, verified=groundedness is not None)
                resolved += 1
                answers.append(text)

        if resolved and self.ledger is not None and hasattr(self.ledger, "bump_answer_version"):
            self.ledger.bump_answer_version()

        total = len(sub_intents)
        self._publish(MultiIntentResolutionEvent(
            turn_id=turn_id, total_sub_intents=total, resolved_count=resolved, suppressed_count=suppressed,
            uncertain_count=uncertain, resolution_recall=resolved / max(1, total - suppressed) if total - suppressed > 0 else 1.0,
            gold_sub_intents=gold_sub_intent_count(sub_gold_map), timestamp=self._now(),
        ))
        if not state["first_emitted"] and uncertainty_lines:
            self._first_token(turn_id, uncertainty_lines[0], state)

        output = " ".join(answers)
        retrieval_calls = early_calls + (self.sub_executor.retrieval_calls - calls_before) + (pending.retrieval_calls if pending else 0)
        speculation: Dict[str, Any] = {}
        if pending is not None and decision is not None and wait is not None:
            speculation = {
                "reconcile_outcome": decision.outcome.value, "predraft_ready_before_end": wait["before_end"],
                "wasted_tokens": counts["wasted_tokens"], "discarded_predrafts": counts["discarded_predrafts"],
            }
            self._publish(ReconcileCompletedEvent(
                turn_id=turn_id, outcome=decision.outcome.value, reason=decision.reason,
                stands=counts["stands"], refined=counts["refined"], redone=counts["redone"],
                wasted_tokens=counts["wasted_tokens"], predraft_tokens=pending.tokens,
                new_chunk_ids=decision.new_chunk_ids, dropped_chunk_ids=decision.dropped_chunk_ids,
                discarded_claims=counts["discarded_claims"], discarded_predrafts=counts["discarded_predrafts"],
                predraft_ready_before_end=wait["before_end"], predraft_late_ms=round(wait["late_ms"], 3),
                timed_out=wait["timed_out"], predraft_ready_ts_s=pending.ready_at, timestamp=self._now(),
            ))
        self._complete(turn_id, output or " ".join(uncertainty_lines), state, totals, retrieval_calls, speculation)

        citations = extract_cites(" ".join(answers))
        turn_output = TurnOutput(
            sub_queries=[s["text"] for s in sub_intents], answer=output, citations=citations,
            uncertainty=" ".join(uncertainty_lines) or None,
            meta={"answer_version": getattr(self.ledger, "answer_version", None), "retrieval_required": True,
                  "ttft_ms": state["ttft_ms"], "tokens_in": int(totals["tokens_in"]), "tokens_out": int(totals["tokens_out"])},
        )
        return {
            "output": output,
            "latency_ms": (self._now() - state["t_start"]) * 1000.0,
            "resolved_count": resolved,
            "suppressed_count": suppressed,
            "groundedness": sum(groundedness_scores) / len(groundedness_scores) if groundedness_scores else 0.0,
            "chunks": all_chunks,
            "retrieval_calls": retrieval_calls,
            "sub_queries": [s["text"] for s in sub_intents],
            "turn_output": turn_output,
            **({"reconcile_outcome": decision.outcome.value, "reconcile_reason": decision.reason, **counts} if decision is not None else {}),
        }

    def _reconciled_sub(
        self, rec: Any, s_text: str, chunks: List[Any], turn_id: str, state: Dict[str, Any],
        totals: Dict[str, float], counts: Dict[str, int],
    ) -> Tuple[str, Optional[float]]:
        """stands / refined: reuse the verified pre-draft claims (minus any citing a chunk that left the
        evidence); draft and verify only the new chunks. Claims enter the answer only here."""
        pre = rec.pending
        kept, discarded = kept_claims(pre, rec.dropped_ids)
        counts["stands"] += len(kept)
        counts["discarded_claims"] += discarded
        kept_text = " ".join(_cited(c) for c in kept)
        supported = max(0, pre.verified.supported - discarded)
        total = max(0, pre.verified.total - discarded)
        self._publish(VerificationEvent(  # reconcile is the single point where pre-drafted claims enter the answer
            turn_id=turn_id, grounded=total > 0 and supported == total, groundedness_score=supported / total if total else 0.0,
            supported_claims=[f"predraft:{i}" for i in range(supported)],
            unsupported_claims=[f"predraft:{i}" for i in range(supported, total)], timestamp=self._now(),
        ))
        if kept_text:
            self._first_token(turn_id, kept_text, state)
        new_chunks = [c for c in chunks if chunk_id_of(c, "") in rec.new_chunk_set]
        new_text, new_supported, new_total = "", 0, 0
        if new_chunks:  # refined: only claims for the new chunks are drafted and verified
            charge(self.clock, self.latency.draft_s + self.latency.verify_s)
            draft = self._draft(s_text, new_chunks, totals)
            new_text, _ = self._verify(draft, new_chunks, turn_id)
            new_total, new_supported = len(extract_cites_claims(draft)), len(extract_cites_claims(new_text))
            counts["refined"] += new_supported
            if new_text and not state["first_emitted"]:
                self._first_token(turn_id, new_text, state)
        text = " ".join(t for t in (kept_text, new_text) if t)
        all_total = total + new_total
        return text, ((supported + new_supported) / all_total if all_total else None)
