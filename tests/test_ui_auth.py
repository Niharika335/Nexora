"""Trace UI auth (HTTP Basic on /ui/* and /api/*) and the endpoints the rewritten page uses."""

import base64
import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from slrag.config import DEFAULT_CONFIG, TelemetryConfig
from slrag.replay.baselines import build_index, mode_config
from slrag.server.app import create_app
from slrag.telemetry.bus import TelemetryBus


def basic(user, password):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


GOOD = basic("admin", "slrag")


@pytest.fixture(scope="module")
def index():
    return build_index(mode_config("ours").app)


def make_app(tmp_path, index):
    bus = TelemetryBus(TelemetryConfig(jsonl_log_path=str(tmp_path / "telemetry.jsonl")))
    app = create_app(config=DEFAULT_CONFIG, engine=index, bus=bus, out_dir=tmp_path)
    app.state.replay_index = index
    return app


@pytest.mark.parametrize("path", ["/ui/trace", "/ui/fields.json", "/api/summary", "/api/scenarios", "/api/experiments"])
def test_ui_and_api_require_basic_auth(tmp_path, index, path):
    with TestClient(make_app(tmp_path, index)) as client:
        for headers in ({}, basic("admin", "wrong"), basic("someone", "slrag"), {"Authorization": "Basic not-base64!"}):
            res = client.get(path, headers=headers)
            assert res.status_code == 401
            assert res.headers["www-authenticate"] == 'Basic realm="SLRAG Trace UI"'
        assert client.get(path, headers=GOOD).status_code in (200, 404)  # 404 only for a missing summary.json


def test_health_and_websockets_are_exempt(tmp_path, index):
    with TestClient(make_app(tmp_path, index)) as client:
        assert client.get("/health").status_code == 200
        with client.websocket_connect("/ws/telemetry"):
            pass


def test_env_overrides_credentials(tmp_path, index, monkeypatch):
    monkeypatch.setenv("SLRAG_UI_USERNAME", "ops")
    monkeypatch.setenv("SLRAG_UI_PASSWORD", "s3cret-é")
    with TestClient(make_app(tmp_path, index)) as client:
        assert client.get("/api/scenarios", headers=GOOD).status_code == 401
        assert client.get("/api/scenarios", headers=basic("ops", "s3cret-é")).status_code == 200


def test_experiments_endpoint(tmp_path, index):
    with TestClient(make_app(tmp_path, index), headers=GOOD) as client:
        assert client.get("/api/experiments").json() == {}
        data = {"split": "test", "experiments": {"A3": {"label": "Speculation", "arms": {"off": {"ttft_p50": 490.0}}}}}
        (tmp_path / "final_experiments.json").write_text(json.dumps(data), encoding="utf-8")
        assert client.get("/api/experiments").json() == data


def test_run_returns_metrics_and_events_and_the_session_can_be_reloaded(tmp_path, index):
    import time

    with TestClient(make_app(tmp_path, index), headers=GOOD) as client:
        b1 = client.post("/api/run", json={"scenario_id": "compound_03", "mode": "b1", "playback": False, "return_events": True}).json()
        assert b1["mode"] == "b1" and len(b1["event_list"]) == b1["events"] > 0
        assert set(b1["metrics"]) >= {"early_retrieval_rate", "groundedness", "g2_pass"}
        run = client.post("/api/run", json={"scenario_id": "compound_03", "speed": 0}).json()
        assert "event_list" not in run and isinstance(run["metrics"]["g2_pass"], bool)
        deadline = time.time() + 5
        res = client.get(f"/api/session/{run['session_id']}")
        while res.status_code != 200 and time.time() < deadline:
            time.sleep(0.1)
            res = client.get(f"/api/session/{run['session_id']}")
        events = [json.loads(line) for line in res.text.splitlines()]
        assert events and {e["session_id"] for e in events} == {run["session_id"]}
        assert client.get("/api/session/nope").status_code == 404
