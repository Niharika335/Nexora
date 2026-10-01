"""FastAPI server exposing /health, /ws/stream, /ws/telemetry, /metrics, /debug/search, the trace UI
(/ui/trace) and its /api/* endpoints.

`uvicorn slrag.server.app:app` serves a default app over the replay corpora (built on first access)."""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from slrag.config import AppConfig, DEFAULT_CONFIG
from slrag.contracts.events import (
    BaseEvent,
    SessionEnd,
    TranscriptChunk,
    UtteranceEnd,
    AnswerDelta,
    Ledger,
    TurnSummary,
)
from slrag.gateway.session import SessionRegistry, GLOBAL_SESSION_REGISTRY
from slrag.pipeline.live import LiveStreamDriver
from slrag.retrieval.engine import HybridRetrievalEngine
from slrag.retrieval.instrumented import instrumented_search
from slrag.telemetry.bus import TelemetryBus, GLOBAL_BUS
from slrag.telemetry.metrics import MetricsCollector, GLOBAL_METRICS

logger = logging.getLogger(__name__)


def _merge_chunk(session_ctx: Any, chunk: TranscriptChunk) -> Tuple[TranscriptChunk, str]:
    """Merge a chunk (cumulative or delta) into the session buffer; returns (annotated event, new text delta)."""
    delta, buffer_text, mode = session_ctx.normalizer.merge(chunk.text)
    return chunk.model_copy(update={"buffer_text": buffer_text, "merge_mode": mode}), delta


class DebugSearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    top_k: int = 10
    session_id: str = "debug_session"


def create_app(
    config: AppConfig = DEFAULT_CONFIG,
    engine: Optional[HybridRetrievalEngine] = None,
    bus: Optional[TelemetryBus] = None,
    registry: Optional[SessionRegistry] = None,
    metrics: Optional[MetricsCollector] = None,
    turn_engine: Optional[Any] = None,
    out_dir: Optional[Path] = None,
) -> FastAPI:
    """Create and configure the FastAPI application using modern lifespan handlers."""

    telemetry_bus = bus or GLOBAL_BUS
    session_reg = registry or GLOBAL_SESSION_REGISTRY
    retrieval_eng = engine or HybridRetrievalEngine(config)
    metrics_coll = metrics or GLOBAL_METRICS

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        telemetry_bus.start()
        await session_reg.start()
        yield
        await session_reg.stop()
        await telemetry_bus.close()

    app = FastAPI(
        title="SL-RAG Streaming Gateway & Telemetry Service",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.config = config
    app.state.engine = retrieval_eng
    app.state.bus = telemetry_bus
    app.state.registry = session_reg
    app.state.metrics = metrics_coll
    app.state.turn_engine = turn_engine
    # The streaming controller runs on live ingestion whenever there is a pipeline to answer with.
    app.state.live = LiveStreamDriver(retrieval_eng, config, telemetry_bus, predrafter=turn_engine) if turn_engine else None
    app.state.start_time = time.time()

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        uptime = round(time.time() - app.state.start_time, 2)
        return {
            "status": "healthy",
            "uptime_seconds": uptime,
            "active_sessions": app.state.registry.active_sessions_count,
            "cfg_hash": app.state.config.cfg_hash,
            "indexed_chunks": len(app.state.engine.chunks_map),
        }

    @app.get("/metrics")
    async def get_metrics(format: str = Query("prometheus", pattern="^(prometheus|json)$")):
        """Metrics endpoint."""
        if format == "json":
            return app.state.metrics.get_snapshot().model_dump()
        return PlainTextResponse(app.state.metrics.get_prometheus_metrics())

    @app.post("/debug/search")
    async def debug_search(req: DebugSearchRequest):
        """Ad-hoc search debug endpoint."""
        if not app.state.engine.chunks_map:
            raise HTTPException(status_code=400, detail="Corpus index is empty.")

        turn_id = f"debug_turn_{int(time.time() * 1000)}"
        results = await instrumented_search(
            engine=app.state.engine,
            query=req.query,
            turn_id=turn_id,
            session_id=req.session_id,
            mode=req.mode,  # type: ignore
            top_k=req.top_k,
            bus=app.state.bus,
            metrics=app.state.metrics,
        )

        return {
            "query": req.query,
            "mode": req.mode,
            "top_k": req.top_k,
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "score": r.score,
                    "rank": r.rank,
                    "section_title": r.section_title,
                    "doc_id": r.doc_id,
                    "text_preview": (r.text[:150] + "...") if len(r.text) > 150 else r.text,
                    "source_scores": r.source_scores,
                }
                for r in results
            ],
        }

    @app.websocket("/ws/telemetry")
    async def websocket_telemetry(websocket: WebSocket):
        """Telemetry WebSocket subscription endpoint broadcasting all system events."""
        await websocket.accept()
        sub_queue = await app.state.bus.register_subscriber()
        try:
            while True:
                event = await sub_queue.get()
                await websocket.send_text(event.model_dump_json())
                sub_queue.task_done()
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"Error in telemetry websocket: {e}")
        finally:
            await app.state.bus.unregister_subscriber(sub_queue)

    @app.websocket("/ws/stream")
    async def websocket_stream(websocket: WebSocket):
        """Bidirectional streaming gateway endpoint for user transcripts and answer deltas."""
        await websocket.accept()
        session_id = websocket.query_params.get("session_id", "ws_default_session")
        session_ctx = await app.state.registry.get_or_create(session_id)

        try:
            while True:
                msg_text = await websocket.receive_text()
                raw_data = json.loads(msg_text)
                
                # Normalize keys
                normalized = session_ctx.normalizer.normalize_dict_keys(raw_data)
                evt_type = normalized.get("event_type", "transcript_chunk")
                seq = int(normalized.get("seq", 0))

                if evt_type == "transcript_chunk":
                    chunk_evt = TranscriptChunk(
                        session_id=session_id,
                        seq=seq,
                        text=normalized.get("text", ""),
                        is_final=bool(normalized.get("is_final", False)),
                        speaker=normalized.get("speaker", "user"),
                        ts_s=normalized.get("ts_s"),
                        cfg_hash=app.state.config.cfg_hash,
                    )
                    # Pass through reorder buffer, then merge into the utterance buffer in seq order
                    ordered_events = session_ctx.reorder_buffer.push(chunk_evt)
                    for evt in ordered_events:
                        merged, delta = _merge_chunk(session_ctx, evt)
                        await app.state.bus.emit(merged)
                        app.state.metrics.record_event("transcript_chunk")
                        if app.state.live:
                            await app.state.live.on_chunk(session_ctx, delta)

                elif evt_type == "utterance_end":
                    # Flush reorder buffer at utterance end so the merged buffer is complete
                    flushed = session_ctx.reorder_buffer.flush_all()
                    for evt in flushed:
                        merged, delta = _merge_chunk(session_ctx, evt)
                        await app.state.bus.emit(merged)
                        if app.state.live:
                            await app.state.live.on_chunk(session_ctx, delta)

                    live_turn = app.state.live.pop_turn(session_ctx) if app.state.live else None
                    buffered_text = session_ctx.normalizer.reset_utterance()
                    utt_evt = UtteranceEnd(
                        session_id=session_id,
                        seq=seq,
                        final_text=normalized.get("text", "") or normalized.get("final_text", "") or buffered_text,
                        turn_id=normalized.get("turn_id") or (live_turn.turn_id if live_turn else session_ctx.next_turn_id()),
                        ts_s=normalized.get("ts_s"),
                        cfg_hash=app.state.config.cfg_hash,
                    )

                    await app.state.bus.emit(utt_evt)
                    app.state.metrics.record_event("utterance_end")

                    # If turn_engine is attached, process turn end-to-end
                    if app.state.turn_engine:
                        async with session_ctx.lock:
                            prefetched = await app.state.live.resolve(live_turn, utt_evt.final_text) if live_turn else None
                            suppressed = bool(live_turn and live_turn.suppressed)
                            pending = live_turn.pending if live_turn else None
                            refine = not suppressed and app.state.turn_engine.should_refine(session_ctx)
                            if pending is not None and (suppressed or refine):
                                app.state.turn_engine.discard_live_pending(
                                    session_ctx, pending, utt_evt.turn_id, "presentation_turn" if suppressed else "refinement_turn")
                                pending = None
                            # process_turn_stream logs every event it yields itself; the other streams leave it to us.
                            engine_logs = False
                            if suppressed:
                                out_stream = app.state.turn_engine.present_turn_stream(session_ctx, utt_evt.final_text, utt_evt.turn_id)
                            elif refine:
                                # Phase 7: a later content turn refines the session answer (delta planner path).
                                out_stream = app.state.turn_engine.refine_turn_stream(session_ctx, utt_evt.final_text, utt_evt.turn_id)
                            else:
                                engine_logs = True
                                out_stream = app.state.turn_engine.process_turn_stream(
                                    session_ctx=session_ctx,
                                    utterance=utt_evt.final_text,
                                    turn_id=utt_evt.turn_id,
                                    prefetched=prefetched,
                                    pending=pending,
                                    early_calls=live_turn.fresh_retrieval_count if live_turn else 0,
                                )
                            async for out_evt in out_stream:
                                await websocket.send_text(out_evt.model_dump_json())
                                if not engine_logs:  # no duplicate telemetry
                                    await app.state.bus.emit(out_evt)
                                app.state.metrics.record_event(out_evt.event_type)
                            if live_turn:
                                live_turn.close(retrieval_required=not suppressed)

                elif evt_type == "session_end":
                    if app.state.live:
                        app.state.live.discard(app.state.live.pop_turn(session_ctx))
                    await app.state.bus.emit(SessionEnd(session_id=session_id, seq=seq, cfg_hash=app.state.config.cfg_hash))
                    await app.state.registry.remove_session(session_id)
                    break

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"Error in stream websocket: {e}", exc_info=True)

    from slrag.server.routes import register_ui_routes  # Phase 9 trace UI

    register_ui_routes(app, out_dir)
    return app


def build_default_app() -> FastAPI:
    """App over the replay corpora with the full live pipeline (what `uvicorn slrag.server.app:app` serves)."""
    from slrag.pipeline.turn_engine import BatchTurnEngine
    from slrag.replay.baselines import build_index

    engine = build_index(DEFAULT_CONFIG)
    turn_engine = BatchTurnEngine(DEFAULT_CONFIG, engine=engine, bus=GLOBAL_BUS, metrics=GLOBAL_METRICS)
    return create_app(config=DEFAULT_CONFIG, engine=engine, bus=GLOBAL_BUS, metrics=GLOBAL_METRICS, turn_engine=turn_engine)


_default_app: Optional[FastAPI] = None


def __getattr__(name: str) -> Any:
    """Module attribute `app`, built lazily so importing create_app does not index the corpus."""
    global _default_app
    if name == "app":
        if _default_app is None:
            _default_app = build_default_app()
        return _default_app
    raise AttributeError(name)
