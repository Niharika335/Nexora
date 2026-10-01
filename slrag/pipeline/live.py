"""Live-ingestion driver: runs the streaming retrieval controller per session for /ws/stream."""
from __future__ import annotations

from typing import Any, List, Optional

from slrag.config import AppConfig, DEFAULT_CONFIG
from slrag.control.controller import RetrievalController
from slrag.control.t2_llm import T2Classifier
from slrag.nlp.lemmas import CorpusVocab
from slrag.pipeline.ledger import ClaimLedger
from slrag.pipeline.streaming import StreamingTurn
from slrag.retrieval.cache import EvidenceCache
from slrag.retrieval.engine import HybridRetrievalEngine, SearchResult
from slrag.telemetry.bus import SessionStampedBus

LIVE_TOP_K = 5  # matches BatchTurnEngine's retrieval depth, so early evidence is interchangeable


class LiveStreamDriver:
    """Feeds each merged transcript delta of a session's current turn to a StreamingTurn.

    State lives on the session: the in-progress StreamingTurn, its turn id, and a per-session
    EvidenceCache (so later turns can reuse evidence too). The session ledger is shared with
    BatchTurnEngine, so T0 presentation rules see the claims already answered.
    """

    def __init__(
        self,
        engine: HybridRetrievalEngine,
        config: AppConfig = DEFAULT_CONFIG,
        bus: Optional[Any] = None,
        predrafter: Optional[Any] = None,
    ):
        self.engine = engine
        self.config = config
        self.bus = bus
        # Phase 8: object with start_live_predraft(stream) (the BatchTurnEngine), used at COMMIT.
        self.predrafter = predrafter
        self.controller = RetrievalController(
            config.controller,
            vocab=CorpusVocab(engine.bm25_index.idf_table) if engine.chunks_map else None,
            t2=T2Classifier(timeout_s=config.controller.t2_timeout_s),
        )

    @property
    def enabled(self) -> bool:
        return self.config.controller.mode != "batch"

    def _retrieve(self, query: str) -> List[SearchResult]:
        return self.engine.search(query, mode="hybrid", top_k=LIVE_TOP_K)

    def _cache(self, session_ctx: Any) -> EvidenceCache:
        if "evidence_cache" not in session_ctx.custom_state:
            session_ctx.custom_state["evidence_cache"] = EvidenceCache(
                ttl_ms=None, bus=self._session_bus(session_ctx), reuse_cos=self.config.cache.reuse_cos,
            )
        return session_ctx.custom_state["evidence_cache"]

    def _session_bus(self, session_ctx: Any) -> Optional[Any]:
        return SessionStampedBus(self.bus, session_ctx.session_id) if self.bus is not None else None

    def _ledger(self, session_ctx: Any) -> ClaimLedger:
        if "ledger" not in session_ctx.custom_state:
            session_ctx.custom_state["ledger"] = ClaimLedger(session_ctx.session_id)
        return session_ctx.custom_state["ledger"]

    def current_turn(self, session_ctx: Any) -> Optional[StreamingTurn]:
        return session_ctx.custom_state.get("live_turn")

    async def on_chunk(self, session_ctx: Any, delta_text: str) -> None:
        if not self.enabled or not delta_text.strip():
            return
        turn = self.current_turn(session_ctx)
        if turn is None:
            ledger = self._ledger(session_ctx)
            refine = self.config.refinement.enabled and ledger.answer_version >= 1
            spec = self.config.speculation
            turn = StreamingTurn(
                session_ctx.next_turn_id(), self.controller, self._retrieve, self._cache(session_ctx),
                bus=self._session_bus(session_ctx), ledger=ledger,
                # Phase 7: once the session has an answer, a content turn is refined at utterance_end,
                # so early raw-utterance retrieval is paused (T0 presentation detection still runs).
                # speculation.mode == off also pauses it (A3 arm "off").
                defer_dispatch=refine or not spec.provisional_on,
                # Phase 8: commit-stage pre-draft (speculation.mode == full and speculation.predraft).
                on_commit=self.predrafter.start_live_predraft if (self.predrafter is not None and spec.predraft_on and not refine) else None,
            )
            turn.session_ctx = session_ctx
            session_ctx.custom_state["live_turn"] = turn
        await turn.on_chunk(delta_text)

    def pop_turn(self, session_ctx: Any) -> Optional[StreamingTurn]:
        return session_ctx.custom_state.pop("live_turn", None)

    async def resolve(self, turn: StreamingTurn, final_text: str) -> Optional[List[SearchResult]]:
        """utterance_end: land outstanding early retrievals and return reusable evidence for the final query, if any."""
        await turn.settle()
        if turn.suppressed:
            return None
        hit = turn.cache.lookup(final_text, turn_id=turn.turn_id)
        if hit is None:
            return None
        hit.entry.used = True
        return list(hit.entry.evidence)

    @staticmethod
    def discard(turn: Optional[StreamingTurn]) -> None:
        """Session ended mid-utterance: cancel in-flight retrievals."""
        if turn is None:
            return
        if turn.pending is not None:
            turn.pending.cancel("session_end")
        for d in turn.dispatches:
            if d.task is not None and not d.task.done():
                d.task.cancel()
