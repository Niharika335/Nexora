"""Streaming turn state machine: per-chunk controller decisions, early retrieval, epochs, stale discard."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

from slrag.contracts.events import ControllerDecisionEvent, SpeculationOutcomeEvent, SpeculativeRetrievalEvent
from slrag.control.controller import ControllerResult, Decision, RetrievalController, TurnState
from slrag.eval.clock import DEFAULT_LATENCY, LatencyModel
from slrag.nlp.lemmas import extract_slots, slot_conflict
from slrag.retrieval.cache import EvidenceCache


@dataclass
class Dispatch:
    retrieval_id: str
    query: str
    trigger: str  # provisional | multi_intent | final
    epoch: int
    dispatched_at: float
    ready_at: float
    source: str = "fresh"
    task: Optional[asyncio.Task] = None
    stale: bool = False
    landed: bool = False
    evidence: Optional[List[Any]] = None  # cache-served evidence (fresh evidence is the task result)


class StreamingTurn:
    """One user turn, fed chunk by chunk.

    Each chunk runs the RetrievalController under the turn lock, then:
      WAIT -> nothing; PROVISIONAL -> dispatch retrieval for Q_t (a new-information dispatch
      bumps the epoch, so older in-flight retrievals become stale); COMMIT -> dispatch for Q_t,
      or per planned sub-query when commit_hint.multi_intent; SUPPRESS -> mark the turn for the
      Presenter path. Results land in the evidence cache unless their epoch is stale, in which
      case they are discarded and a `stale_discard` outcome is logged.
    """

    def __init__(
        self,
        turn_id: str,
        controller: RetrievalController,
        retrieve: Callable[[str], List[Any]],
        cache: EvidenceCache,
        bus: Optional[Any] = None,
        clock: Optional[Any] = None,
        ledger: Any = None,
        planner: Optional[Any] = None,
        latency: LatencyModel = DEFAULT_LATENCY,
        defer_dispatch: bool = False,
        on_commit: Optional[Callable[["StreamingTurn"], Awaitable[Optional[Any]]]] = None,
    ):
        self.state = TurnState(turn_id=turn_id)
        # Phase 8: called once at COMMIT (after the commit retrievals are dispatched) to start the
        # background pre-draft; returns the PendingDraft (or None).
        self.on_commit = on_commit
        self.pending: Optional[Any] = None
        self.session_ctx: Optional[Any] = None  # set by the live driver (the live pre-draft needs the session)
        # Phase 7 refinement turns: decisions are made and logged, but retrieval waits for the delta plan.
        self.defer_dispatch = defer_dispatch
        self.controller = controller
        self.retrieve = retrieve
        self.cache = cache
        self.bus = bus
        self.clock = clock
        self.ledger = ledger
        self.planner = planner
        self.latency = latency
        self.lock = asyncio.Lock()
        self.dispatches: List[Dispatch] = []
        self.decisions: List[ControllerResult] = []
        self.suppressed = False
        self.committed_plan: Optional[Any] = None
        self._counter = 0

    @property
    def turn_id(self) -> str:
        return self.state.turn_id

    def _now(self) -> float:
        return self.clock.time() if self.clock else time.time()

    def _publish(self, event: Any) -> None:
        if self.bus:
            self.bus.publish(event)

    async def on_chunk(self, text: str) -> ControllerResult:
        async with self.lock:
            await self._land_ready()
            self.state.chunks.append(text)
            self.state.buffer = f"{self.state.buffer} {text}".strip()
            result = await self.controller.on_chunk(self.state, self.ledger)
            self.decisions.append(result)
            self._publish(ControllerDecisionEvent(
                turn_id=self.turn_id, timestamp=self._now(), decision=result.decision.value, tier=result.tier,
                reason=result.reason, query=result.query, epoch=self.state.epoch, payload=result.payload,
                chunk_seq=len(self.state.chunks), chunk_text=text,
            ))

            if self.defer_dispatch and result.decision in (Decision.PROVISIONAL, Decision.COMMIT):
                pass
            elif result.decision == Decision.PROVISIONAL and result.query:
                if result.reason in ("new_information", "slot_change"):
                    self._bump_epoch()
                self._dispatch(result.query, "provisional")
            elif result.decision == Decision.COMMIT and result.query:
                if result.commit_hint.get("multi_intent") and self.planner is not None:
                    self.committed_plan = await self.planner.plan(self.state.buffer)
                    for sq in self.committed_plan.sub_queries:
                        self._dispatch(sq.text, "multi_intent")
                else:
                    self._dispatch(result.query, "final")
                if self.on_commit is not None and self.pending is None:
                    self.pending = await self.on_commit(self)
                    if self.pending is not None:
                        self.state.predraft_status = "pending"
            elif result.decision == Decision.SUPPRESS:
                self.suppressed = True
            if result.clause_end and not self.defer_dispatch and not self.suppressed:
                await self._dispatch_completed_clauses()
            return result

    def _covered(self, query: str) -> bool:
        """An existing (in-flight or landed, non-stale) retrieval already serves `query` by the cache reuse rule."""
        if self.cache.peek(query) is not None:
            return True
        emb, slots = self.cache.embed(query), extract_slots(query)
        return any(
            not d.stale and float(np.dot(emb, self.cache.embed(d.query))) >= self.cache.reuse_cos
            and slot_conflict(slots, extract_slots(d.query)) is None
            for d in self.dispatches
        )

    async def _dispatch_completed_clauses(self) -> None:
        """Per-clause retrieval: split the buffer with the same deterministic splitter the final path uses and
        retrieve every completed sub-query not yet covered, so the final path finds it in the evidence cache."""
        if self.planner is None:
            return
        plan = self.planner.plan_sync(self.state.buffer)
        if plan.source == "single":
            return
        for sq in plan.sub_queries:
            if not self._covered(sq.text):
                self._dispatch(sq.text, "clause")

    def _bump_epoch(self) -> None:
        """Supersede in-flight retrievals: anything that has not landed by now is stale."""
        now = self._now()
        self.state.epoch += 1
        # The Phase 8 pre-draft is not cancelled here: new information later in the same utterance is a
        # tail change that reconcile handles per sub-intent (refined / redone). It is cancelled when the
        # turn is superseded (presentation turn, session end) or times out.
        for d in self.dispatches:
            if not d.landed and (d.ready_at > now if self.clock else not (d.task and d.task.done())):
                d.stale = True

    def _dispatch(self, query: str, trigger: str) -> Dispatch:
        self._counter += 1
        rid = f"{self.turn_id}:r{self._counter}"
        now = self._now()
        hit = self.cache.lookup(query, turn_id=self.turn_id)
        if hit is not None:
            d = Dispatch(rid, query, trigger, self.state.epoch, now, max(now, hit.entry.ready_at), source="cache", landed=True,
                         evidence=list(hit.entry.evidence))
            hit.entry.used = True
        else:
            d = Dispatch(rid, query, trigger, self.state.epoch, now, now + self.latency.retrieval_s)
            d.task = asyncio.ensure_future(asyncio.to_thread(self.retrieve, query))
        self.dispatches.append(d)
        self._publish(SpeculativeRetrievalEvent(
            turn_id=self.turn_id, timestamp=now, retrieval_id=rid, query=query, trigger=trigger,
            is_early=True, epoch=d.epoch, source=d.source,
            latency_ms=round((d.ready_at - now) * 1000.0, 3) if self.clock else 0.0,  # modeled latency under a replay clock
        ))
        return d

    async def _land(self, d: Dispatch) -> None:
        """A retrieval completed: discard it if its epoch was superseded, else store it in the cache."""
        chunks = await d.task
        d.landed = True
        if d.stale:
            self._publish(SpeculationOutcomeEvent(
                turn_id=self.turn_id, timestamp=self._now(), outcome="stale_discard", reason="epoch_changed", retrieval_id=d.retrieval_id,
            ))
            return
        self.cache.store(d.retrieval_id, d.query, list(chunks), turn_id=self.turn_id, ready_at=d.ready_at)

    async def _land_ready(self) -> None:
        """Land retrievals that have completed by now (virtual ready_at under a replay clock, task.done() otherwise)."""
        now = self._now()
        for d in self.dispatches:
            if d.landed or d.task is None:
                continue
            if (d.ready_at <= now) if self.clock else d.task.done():
                await self._land(d)

    async def settle(self) -> None:
        """utterance_end: land every outstanding retrieval so the final path can reuse it."""
        async with self.lock:
            for d in self.dispatches:
                if not d.landed and d.task is not None:
                    await self._land(d)

    def close(self, retrieval_required: bool = True) -> Dict[str, int]:
        """Publish the outcome of every early retrieval: used / wasted / false_trigger."""
        counts = {"used": 0, "wasted": 0, "false_trigger": 0, "stale_discard": 0}
        for d in self.dispatches:
            if d.stale:
                counts["stale_discard"] += 1
                continue
            if not retrieval_required:
                outcome, reason = "false_trigger", "turn_needed_no_retrieval"
            elif d.source == "cache" or self.cache.is_used(d.retrieval_id):
                outcome, reason = "used", "evidence_reused"
            else:
                outcome, reason = "wasted", "final_query_diverged"
            counts[outcome] += 1
            self._publish(SpeculationOutcomeEvent(
                turn_id=self.turn_id, timestamp=self._now(), outcome=outcome, reason=reason, retrieval_id=d.retrieval_id,
            ))
        return counts

    def commit_dispatches(self) -> List[Dispatch]:
        return [d for d in self.dispatches if d.trigger in ("final", "multi_intent") and not d.stale]

    async def commit_evidence(self, query: str) -> Optional[List[Any]]:
        """Evidence of the COMMIT-stage retrieval for `query` (the only commit dispatch, or the one with the
        same query); awaits it if still in flight. None if no commit retrieval serves this query."""
        commits = self.commit_dispatches()
        d = next((c for c in commits if c.query == query), commits[0] if len(commits) == 1 else None)
        if d is None:
            return None
        if d.evidence is not None:
            return list(d.evidence)
        if d.task is not None:
            return list(await asyncio.shield(d.task))
        return None

    def commit_ready_at(self) -> Optional[float]:
        commits = self.commit_dispatches()
        return max((d.ready_at for d in commits), default=None)

    @property
    def early_retrieval_count(self) -> int:
        return len(self.dispatches)

    @property
    def fresh_retrieval_count(self) -> int:
        return sum(1 for d in self.dispatches if d.source == "fresh")
