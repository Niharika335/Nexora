"""Trace UI routes (Phase 9): the static page, /api/* endpoints and scenario playback.

GET  /ui/trace                        -> slrag/ui/trace.html (also /ui/* static files, e.g. fields.json)
GET  /api/summary                     -> out/summary.json (+ A3/A4 experiment files next to it), 404 if absent
GET  /api/chunk/{chunk_id}            -> {chunk_id, cite, doc_id, section, text} of an indexed chunk
GET  /api/scenarios                   -> [{id, category, turns, path}] from eval/scenarios and eval/demo
POST /api/run {scenario_id, speed}    -> {session_id}: replays the scenario (ours) and plays its telemetry
                                         onto the bus at `speed` (1 = real time on the replay clock)
GET  /api/trace/{mode}/{scenario_id}  -> recorded trace JSONL (out/traces/<mode>_<scenario_id>.jsonl)
GET  /api/experiments                 -> out/final_experiments.json ({} if absent; scripts/run_experiments.py)
GET  /api/session/{session_id}        -> that session's events from the telemetry log, as JSONL

/ui/* and /api/* require HTTP Basic auth (config ui.username / ui.password, or the env vars
SLRAG_UI_USERNAME / SLRAG_UI_PASSWORD). /ws/* and /health are not protected.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
from pathlib import Path
import secrets
from typing import Any, Dict, List, Optional, Tuple
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
UI_DIR = Path(__file__).resolve().parents[1] / "ui"
SCENARIO_DIRS = (REPO_ROOT / "eval" / "scenarios", REPO_ROOT / "eval" / "demo")


class RunRequest(BaseModel):
    scenario_id: str
    speed: float = 1.0
    mode: str = "ours"
    playback: bool = True  # publish the run's telemetry onto the bus (/ws/telemetry) at `speed`
    return_events: bool = False  # also return the events in the response (e.g. a B1 overlay)


PROTECTED_PREFIXES = ("/ui", "/api")
REALM = 'Basic realm="SLRAG Trace UI"'


def ui_credentials(config: Any) -> Tuple[str, str]:
    ui = getattr(config, "ui", None)
    return (
        os.environ.get("SLRAG_UI_USERNAME") or getattr(ui, "username", "admin"),
        os.environ.get("SLRAG_UI_PASSWORD") or getattr(ui, "password", "slrag"),
    )


def basic_auth_ok(header: Optional[str], username: str, password: str) -> bool:
    if not header or not header.lower().startswith("basic "):
        return False
    try:
        user, _, pw = base64.b64decode(header[6:].strip()).decode("utf-8").partition(":")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return False
    user_ok = secrets.compare_digest(user.encode("utf-8"), username.encode("utf-8"))
    password_ok = secrets.compare_digest(pw.encode("utf-8"), password.encode("utf-8"))
    return user_ok and password_ok


def _load_scenarios() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for base in SCENARIO_DIRS:
        for path in sorted(base.glob("*.jsonl")) if base.exists() else []:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    sc = json.loads(line)
                    sc["_path"] = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
                    out.append(sc)
    return out


def _ts(event: Dict[str, Any]) -> Optional[float]:
    t = event.get("timestamp")
    return float(t) if isinstance(t, (int, float)) else None


def register_ui_routes(app: FastAPI, out_dir: Optional[Path] = None) -> None:
    app.state.out_dir = Path(out_dir) if out_dir else REPO_ROOT / "out"
    app.state.replay_index = None
    app.state.runs = {}

    @app.middleware("http")
    async def basic_auth(request: Request, call_next):
        """HTTP Basic auth on /ui/* and /api/* (HTTP only: WebSocket routes never reach this middleware)."""
        path = request.url.path
        if any(path == p or path.startswith(p + "/") for p in PROTECTED_PREFIXES):
            username, password = ui_credentials(app.state.config)
            if not basic_auth_ok(request.headers.get("authorization"), username, password):
                return Response("Authentication required", status_code=401, headers={"WWW-Authenticate": REALM})
        return await call_next(request)

    @app.get("/ui/trace", include_in_schema=False)
    async def trace_page():
        return FileResponse(UI_DIR / "trace.html", media_type="text/html")

    @app.get("/api/summary")
    async def api_summary():
        path = app.state.out_dir / "summary.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="out/summary.json not found: run `slrag replay ... --mode ours` first")
        summary = json.loads(path.read_text(encoding="utf-8"))
        experiments = summary.setdefault("experiments", {})
        for name in ("A3", "A4"):  # experiment results written next to summary.json
            extra = app.state.out_dir / f"{name.lower()}.json"
            if name not in experiments and extra.exists():
                experiments[name] = json.loads(extra.read_text(encoding="utf-8"))
        return summary

    def _index() -> Any:
        if app.state.replay_index is None:
            from slrag.replay.baselines import build_index, mode_config

            app.state.replay_index = build_index(mode_config("ours").app)
        return app.state.replay_index

    @app.get("/api/chunk/{chunk_id:path}")
    async def api_chunk(chunk_id: str):
        chunk = app.state.engine.chunks_map.get(chunk_id)
        if chunk is None:
            chunk = (await asyncio.to_thread(_index)).chunks_map.get(chunk_id)
        if chunk is None:
            raise HTTPException(status_code=404, detail=f"chunk '{chunk_id}' is not indexed")
        return {
            "chunk_id": chunk_id, "cite": f"[{chunk_id}]", "doc_id": chunk.doc_id,
            "section": chunk.metadata.get("section_title", ""), "text": chunk.text,
        }

    @app.get("/api/scenarios")
    async def api_scenarios():
        return [
            {"id": sc["scenario_id"], "category": sc.get("category", ""), "turns": 2 if sc.get("follow_up") else 1, "path": sc["_path"]}
            for sc in _load_scenarios()
        ]

    @app.post("/api/run")
    async def api_run(req: RunRequest):
        sc = next((s for s in _load_scenarios() if s["scenario_id"] == req.scenario_id), None)
        if sc is None:
            raise HTTPException(status_code=404, detail=f"unknown scenario '{req.scenario_id}'")
        from slrag.eval.clock import ReplayClock
        from slrag.eval.runner import InMemoryBus, turn_ids
        from slrag.replay.baselines import build_turn_engine

        index = await asyncio.to_thread(_index)
        bus, clock = InMemoryBus(), ReplayClock()
        engine = build_turn_engine(req.mode, index, bus, clock=clock)
        ids = turn_ids(sc)
        sub_gold = {s["id"]: s.get("gold_chunk_ids", []) for s in sc.get("sub_intents", [])}
        await engine.execute_turn(sc["query"], turn_id=ids[0], sub_gold_map=sub_gold,
                                  is_unanswerable_ground_truth=sc.get("is_unanswerable", False))
        if sc.get("follow_up"):
            clock.advance(2.0)
            await engine.execute_turn(sc["follow_up"]["text"], turn_id=ids[1],
                                      expected_chunk_ids=sc["follow_up"].get("delta_gold_chunk_ids", []))
        session_id = f"ui_{req.mode}_{req.scenario_id}_{uuid.uuid4().hex[:6]}" if req.mode != "ours" else f"ui_{req.scenario_id}_{uuid.uuid4().hex[:6]}"
        events = [e.model_copy(update={"session_id": session_id}) for e in bus.events]
        from slrag.replay.metrics import MetricCalculator

        m = MetricCalculator.calculate_all(events)  # the harness's own metric code, for the run history
        metrics = {k: m.get(k) for k in ("early_retrieval_rate", "groundedness", "recall@10", "ttft_p50", "trace_coverage")}
        metrics["g2_pass"] = m.get("early_retrieval_rate") is not None and m["early_retrieval_rate"] >= 0.80

        async def play() -> None:
            last: Optional[float] = None
            for ev in events:
                t = _ts(ev.model_dump())
                if req.speed > 0 and t is not None and last is not None and t > last:
                    await asyncio.sleep(min(t - last, 5.0) / req.speed)
                if t is not None:
                    last = t
                app.state.bus.publish(ev)

        if req.playback:
            app.state.runs[session_id] = asyncio.create_task(play())
        body: Dict[str, Any] = {"session_id": session_id, "scenario_id": req.scenario_id, "mode": req.mode,
                                "events": len(events), "turn_ids": ids, "metrics": metrics}
        if req.return_events:
            body["event_list"] = [e.model_dump(mode="json") for e in events]
        return body

    @app.get("/api/experiments")
    async def api_experiments():
        path = app.state.out_dir / "final_experiments.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    @app.get("/api/session/{session_id}")
    async def api_session(session_id: str):
        """Events of one session from the telemetry log (what /ws/telemetry streamed), as JSONL."""
        log = getattr(app.state.bus, "log_path", None)
        lines: List[str] = []
        if log is not None and Path(log).exists():
            for line in Path(log).read_text(encoding="utf-8").splitlines():
                if f'"session_id":"{session_id}"' in line.replace(" ", ""):
                    lines.append(line)
        if not lines:
            raise HTTPException(status_code=404, detail=f"no telemetry for session '{session_id}'")
        return PlainTextResponse("\n".join(lines) + "\n", media_type="application/x-ndjson")

    @app.get("/api/trace/{mode}/{scenario_id}")
    async def api_trace(mode: str, scenario_id: str):
        path = app.state.out_dir / "traces" / f"{mode}_{scenario_id}.jsonl"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"no recorded {mode} trace for '{scenario_id}' (replay with --trace-out out/traces)")
        return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="application/x-ndjson")

    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")
