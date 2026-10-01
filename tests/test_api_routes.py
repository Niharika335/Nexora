"""Phase 9 /api/* routes: scenarios, run (playback onto /ws/telemetry's bus), chunk click-through, traces."""

import json
from pathlib import Path
import re
import time

import pytest
from starlette.testclient import TestClient

from slrag.config import DEFAULT_CONFIG, TelemetryConfig
from slrag.eval.runner import ReplayRunner
from slrag.replay.baselines import build_index, mode_config
from slrag.server.app import create_app
from slrag.telemetry.bus import TelemetryBus

ROOT = Path(__file__).resolve().parent.parent
AUTH = {"Authorization": "Basic YWRtaW46c2xyYWc="}  # admin:slrag (config.yaml ui.*)


@pytest.fixture(scope="module")
def index():
    return build_index(mode_config("ours").app)


def client_for(tmp_path, index):
    bus = TelemetryBus(TelemetryConfig(jsonl_log_path=str(tmp_path / "telemetry.jsonl")))
    app = create_app(config=DEFAULT_CONFIG, engine=index, bus=bus, out_dir=tmp_path)
    app.state.replay_index = index
    return TestClient(app, headers=AUTH)


def test_api_scenarios_lists_every_scenario(tmp_path, index):
    with client_for(tmp_path, index) as client:
        scenarios = client.get("/api/scenarios").json()
    ids = {s["id"] for s in scenarios}
    assert len(scenarios) == 90 and {"compound_03", "late_03", "unanswerable_01"} <= ids
    late = next(s for s in scenarios if s["id"] == "late_03")
    assert late["turns"] == 2 and late["category"] == "late_detail" and late["path"].endswith("late_test.jsonl")


def test_api_run_plays_telemetry_and_every_cite_resolves(tmp_path, index):
    log = tmp_path / "telemetry.jsonl"
    with client_for(tmp_path, index) as client:
        run = client.post("/api/run", json={"scenario_id": "late_03", "speed": 0}).json()
        assert run["session_id"].startswith("ui_late_03_") and run["turn_ids"] == ["late_03:t1", "late_03:t2"] and run["events"] > 0
        time.sleep(0.3)  # playback task publishes onto the bus
        cites = set()
        events = []
        deadline = time.time() + 5
        while time.time() < deadline:
            events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
            if len([e for e in events if e.get("session_id") == run["session_id"]]) >= run["events"]:
                break
            time.sleep(0.1)
        mine = [e for e in events if e.get("session_id") == run["session_id"]]
        assert len(mine) == run["events"]
        assert {"answer_version_transition", "plan_completed", "answer_chunk", "controller_decision"} <= {e["event_type"] for e in mine}
        for e in mine:
            if e["event_type"] == "answer_chunk":
                cites.update(e["cites"])
        assert cites
        for cite in cites:  # citation click-through returns the exact stored chunk
            chunk = client.get(f"/api/chunk/{cite}").json()
            assert chunk["chunk_id"] == cite and chunk["text"] == index.chunks_map[cite].text
            assert chunk["doc_id"] == index.chunks_map[cite].doc_id and chunk["cite"] == f"[{cite}]"
        assert client.get("/api/chunk/no-such-chunk").status_code == 404
        assert client.post("/api/run", json={"scenario_id": "nope"}).status_code == 404


def test_api_trace_serves_recorded_baseline_traces(tmp_path, index):
    with client_for(tmp_path, index) as client:
        assert client.get("/api/trace/b1/compound_03").status_code == 404
        ReplayRunner(str(ROOT / "eval" / "scenarios" / "test.jsonl"), mode="b1", out_path=str(tmp_path / "b1.jsonl"),
                     category="compound", trace_out=str(tmp_path / "traces")).run(generate_report=False)
        res = client.get("/api/trace/b1/compound_03")
        assert res.status_code == 200
        events = [json.loads(line) for line in res.text.splitlines()]
        assert events and {e["turn_id"] for e in events} == {"compound_03"}
        assert any(e["event_type"] == "turn_complete" for e in events)


def test_live_session_telemetry_is_session_stamped_and_logged_once(tmp_path):
    """The UI filters by session: live controller / cache / pre-draft events must carry the session id,
    and nothing the gateway streams may be logged twice."""
    from test_live_stream import _app, _send_turn

    log = tmp_path / "live.jsonl"
    with TestClient(_app(log)) as client:
        with client.websocket_connect("/ws/stream?session_id=ui_live") as ws:
            out, seq = _send_turn(ws, ["Explain how Raft leader election", "and log replication work", "across a majority quorum"], 1)
            _send_turn(ws, ["What are the weather conditions", "on Mars right now?"], seq)
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    turn = out[-1]["turn_id"]
    mine = [e for e in events if e.get("turn_id") == turn or e.get("active_turn_id") == turn]
    assert mine and {e["session_id"] for e in mine} == {"ui_live"}
    assert {"controller_decision", "speculative_retrieval", "speculative_cache", "draft_completed"} <= {e["event_type"] for e in mine}
    ids = [e["event_id"] for e in events]
    assert len(ids) == len(set(ids)), "an event was logged twice"
    assert len([e for e in events if e["event_type"] == "turn_summary"]) == 2  # one per turn, incl. the insufficient one


def test_uvicorn_entrypoint_exposes_a_module_level_app(monkeypatch):
    import slrag.server.app as server

    built = []
    monkeypatch.setattr(server, "build_default_app", lambda: built.append(1) or "app-object")
    monkeypatch.setattr(server, "_default_app", None)
    assert server.app == "app-object" and server.app == "app-object" and built == [1]  # built once, lazily
