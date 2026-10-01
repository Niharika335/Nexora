"""Commit-stage pre-drafting (Phase 8).

At COMMIT, while the user is still speaking, a background asyncio.Task runs the draft pipeline
on the commit-time evidence: per sub-intent sufficiency gate -> drafter -> verifier. The result
is held in a PendingDraft: nothing is written to the ledger and nothing is emitted. At
utterance_end the turn engine reconciles it against the final evidence (slrag.answer.reconcile).

The pipeline is generic: engines supply the evidence / gate / draft / verify callables, so the
replay TurnEngine and the live /ws/stream BatchTurnEngine run the same code. Drafts run in
worker threads under a semaphore (draft_concurrency), so the event loop and the main turn path
are never blocked.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np

from slrag.contracts.events import Claim, DraftCompletedEvent
from slrag.eval.clock import DEFAULT_LATENCY, LatencyModel
from slrag.nlp.lemmas import extract_slots
from slrag.retrieval.cache import chunk_id_of, default_embed

# evidence_fn(sub_id, sub_text) -> chunks (awaitable); gate_fn(sub_text, chunks) -> (sufficient, score)
# draft_fn(sub_text, chunks) -> DraftResult (runs in a worker thread)
# verify_fn(draft_result, chunks) -> VerifiedDraft (runs on the loop; must not publish)
EvidenceFn = Callable[[str, str], Awaitable[List[Any]]]
GateFn = Callable[[str, List[Any]], Tuple[bool, Optional[float]]]


@dataclass
class DraftResult:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    raw: Any = None  # engine-specific (e.g. raw claims for the live pipeline)


@dataclass
class VerifiedDraft:
    verified_text: str
    claims: List[Claim]  # claims that passed the verifier, with their cites
    supported: int
    total: int
    checks: List[Any] = field(default_factory=list)  # engine-specific verification records, emitted at reconcile

    @property
    def groundedness(self) -> float:
        return self.supported / self.total if self.total else 0.0


@dataclass
class SubDraft:
    sub_intent_id: str
    text: str
    evidence: List[Any] = field(default_factory=list)
    sufficient: bool = False
    gate_score: Optional[float] = None
    draft: Optional[DraftResult] = None
    verified: Optional[VerifiedDraft] = None
    ready_ts_s: Optional[float] = None

    @property
    def evidence_ids(self) -> List[str]:
        return [chunk_id_of(c, str(i)) for i, c in enumerate(self.evidence)]

    @property
    def tokens(self) -> int:
        return (self.draft.tokens_in + self.draft.tokens_out) if self.draft else 0


@dataclass
class PendingDraft:
    """Held pre-draft of one turn (spec: PendingDraft). Claims are verified but not yet in the ledger."""
    turn_id: str
    committed_text: str
    committed_emb: np.ndarray
    committed_slots: Dict[str, str]
    epoch: int
    sub_drafts: Dict[str, SubDraft] = field(default_factory=dict)
    task: Optional[asyncio.Task] = None
    commit_ts_s: float = 0.0
    ready_at: Optional[float] = None  # virtual ready time of the drafts under a replay clock
    evidence_ready_at: Optional[float] = None  # virtual time the commit evidence is available
    evidence_event: Optional[asyncio.Event] = None
    retrieval_calls: int = 0
    cancelled: bool = False
    cancel_reason: Optional[str] = None  # epoch_changed | timeout | presentation_turn | session_end
    error: Optional[str] = None

    @property
    def tokens_in(self) -> int:
        return sum(s.draft.tokens_in for s in self.sub_drafts.values() if s.draft)

    @property
    def tokens_out(self) -> int:
        return sum(s.draft.tokens_out for s in self.sub_drafts.values() if s.draft)

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def llm_calls(self) -> int:
        return sum(1 for s in self.sub_drafts.values() if s.draft)

    @property
    def cost(self) -> float:
        return sum(s.draft.cost for s in self.sub_drafts.values() if s.draft)

    @property
    def claims(self) -> List[Claim]:
        return [c for s in self.sub_drafts.values() if s.verified for c in s.verified.claims]

    @property
    def done(self) -> bool:
        return self.task is not None and self.task.done()

    def cancel(self, reason: str = "cancelled") -> None:
        if not self.cancelled:
            self.cancelled, self.cancel_reason = True, reason
        if self.task is not None and not self.task.done():
            self.task.cancel()

    async def wait_evidence(self, timeout_s: float) -> bool:
        """Wait (wall clock) until every sub-intent's commit-time evidence is gathered (before any draft)."""
        if self.evidence_event is None or self.cancelled:
            return False
        try:
            await asyncio.wait_for(self.evidence_event.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            self.cancel("timeout")
            return False
        return not self.cancelled and self.error is None

    async def wait(self, timeout_s: float) -> bool:
        """Wait (wall clock) for the pipeline; False on timeout / cancellation / error (the task is cancelled)."""
        if self.task is None or self.cancelled:
            return False
        try:
            await asyncio.wait_for(asyncio.shield(self.task), timeout=timeout_s)
        except asyncio.TimeoutError:
            self.cancel("timeout")
            return False
        except asyncio.CancelledError:
            self.cancel(self.cancel_reason or "cancelled")
            return False
        except Exception as exc:  # a failed pre-draft never fails the turn
            self.error = repr(exc)
            return False
        return not self.cancelled


class Predrafter:
    def __init__(
        self,
        draft_fn: Callable[[str, List[Any]], DraftResult],
        verify_fn: Callable[[DraftResult, List[Any]], VerifiedDraft],
        gate_fn: GateFn,
        bus: Optional[Any] = None,
        clock: Optional[Any] = None,
        latency: LatencyModel = DEFAULT_LATENCY,
        concurrency: int = 3,
        embed: Callable[[str], np.ndarray] = default_embed,
    ):
        self.draft_fn = draft_fn
        self.verify_fn = verify_fn
        self.gate_fn = gate_fn
        self.bus = bus
        self.clock = clock
        self.latency = latency
        self.concurrency = max(1, concurrency)
        self.embed = embed
        self.draft_lock = threading.Lock()  # drafters keep per-call usage state; one call at a time per drafter

    def _now(self) -> float:
        return self.clock.time() if self.clock else time.time()

    def start(
        self,
        turn_id: str,
        committed_text: str,
        sub_queries: List[Tuple[str, str]],
        evidence_fn: EvidenceFn,
        epoch: int = 0,
        evidence_ready_at: Optional[float] = None,
    ) -> PendingDraft:
        """Create the PendingDraft and launch the background task; returns immediately."""
        now = self._now()
        pending = PendingDraft(
            turn_id=turn_id, committed_text=committed_text, committed_emb=self.embed(committed_text),
            committed_slots=extract_slots(committed_text), epoch=epoch, commit_ts_s=now,
            sub_drafts={sid: SubDraft(sid, text) for sid, text in sub_queries},
        )
        if self.clock is not None:
            # Virtual timeline: evidence lands, then drafts run concurrently in batches of `concurrency`, then verify.
            batches = -(-len(sub_queries) // self.concurrency) if sub_queries else 0
            start = max(now, evidence_ready_at if evidence_ready_at is not None else now)
            pending.evidence_ready_at = start
            pending.ready_at = start + batches * self.latency.draft_s + self.latency.verify_s
        pending.evidence_event = asyncio.Event()
        pending.task = asyncio.ensure_future(self._run(pending, evidence_fn))
        return pending

    async def _run(self, pending: PendingDraft, evidence_fn: EvidenceFn) -> PendingDraft:
        sem = asyncio.Semaphore(self.concurrency)
        remaining = [len(pending.sub_drafts)]

        async def one(sub: SubDraft) -> None:
            try:
                sub.evidence = list(await evidence_fn(sub.sub_intent_id, sub.text))
            finally:
                remaining[0] -= 1
                if remaining[0] == 0 and pending.evidence_event is not None:
                    pending.evidence_event.set()
            sub.sufficient, sub.gate_score = self.gate_fn(sub.text, sub.evidence)
            if sub.sufficient and sub.evidence:
                async with sem:
                    sub.draft = await asyncio.to_thread(self._draft, sub.text, sub.evidence)
                sub.verified = self.verify_fn(sub.draft, sub.evidence)
            sub.ready_ts_s = pending.ready_at if pending.ready_at is not None else self._now()
            if self.bus is not None:
                self.bus.publish(DraftCompletedEvent(
                    turn_id=pending.turn_id, timestamp=sub.ready_ts_s, sub_intent_id=sub.sub_intent_id, speculative=True,
                    ready_ts_s=sub.ready_ts_s, sufficient=sub.sufficient,
                    claims_count=len(sub.verified.claims) if sub.verified else 0,
                    tokens_in=sub.draft.tokens_in if sub.draft else 0, tokens_out=sub.draft.tokens_out if sub.draft else 0,
                ))

        if not pending.sub_drafts and pending.evidence_event is not None:
            pending.evidence_event.set()
        try:
            await asyncio.gather(*(one(sub) for sub in pending.sub_drafts.values()))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            pending.error = repr(exc)
            if pending.evidence_event is not None:
                pending.evidence_event.set()
            raise
        return pending

    def _draft(self, text: str, chunks: List[Any]) -> DraftResult:
        with self.draft_lock:
            return self.draft_fn(text, chunks)
