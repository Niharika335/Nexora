"""Phase 9 trace UI rendering pipeline without a browser: the fixture telemetry parses into the event
contracts and carries every field the page reads; /ui/trace and /api/summary are served."""

import inspect
import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

import slrag.contracts.events as events_module
from slrag.config import DEFAULT_CONFIG, TelemetryConfig
from slrag.contracts.events import BaseEvent
from slrag.corpus.loader import CorpusLoader
from slrag.retrieval.engine import HybridRetrievalEngine
from slrag.server.app import create_app
from slrag.telemetry.bus import TelemetryBus

ROOT = Path(__file__).resolve().parent.parent
AUTH = {"Authorization": "Basic YWRtaW46c2xyYWc="}  # admin:slrag (config.yaml ui.*)
FIXTURE = ROOT / "tests" / "fixtures" / "trace_10_events.jsonl"
MANIFEST = json.loads((ROOT / "slrag" / "ui" / "fields.json").read_text(encoding="utf-8"))

EVENT_MODELS = {
    cls.model_fields["event_type"].default: cls
    for _, cls in inspect.getmembers(events_module, inspect.isclass)
    if issubclass(cls, BaseEvent) and "event_type" in cls.model_fields
}


def get(obj, path):
    for part in path.split("."):
        if isinstance(obj, list):
            obj = obj[0] if obj else None
        if not isinstance(obj, dict) or part not in obj:
            return KeyError
        obj = obj[part]
    return obj


def test_fixture_events_parse_into_the_contracts_with_every_rendered_field():
    events = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(events) == 10
    for e in events:
        model = EVENT_MODELS[e["event_type"]]
        parsed = model.model_validate(e)  # the page renders exactly what the contracts carry
        assert parsed.event_type == e["event_type"]
        spec = MANIFEST["events"][e["event_type"]]
        for path in spec["required"]:
            assert get(e, path) is not KeyError, f"{e['event_type']}.{path}"
    # The same pipeline the page runs: turn grouping, early retrieval before utterance_end, ledger v1 from ops.
    assert {e["turn_id"] for e in events} == {"demo_1"}
    end = next(e["timestamp"] for e in events if e["event_type"] == "utterance_final")
    assert all(e["timestamp"] < end for e in events if e["event_type"] == "speculative_retrieval" and e["is_early"])
    delta = next(e for e in events if e["event_type"] == "answer_delta")
    assert [op["op"] for op in delta["ops"]] == ["add"] and set(delta["hashes"]) == {"c1"}


def _client(tmp_path, summary=None):
    engine = HybridRetrievalEngine(DEFAULT_CONFIG)
    engine.index_documents(CorpusLoader.load_file(ROOT / "data" / "sample_corpus.json"))
    if summary is not None:
        (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    bus = TelemetryBus(TelemetryConfig(jsonl_log_path=str(tmp_path / "t.jsonl")))
    return TestClient(create_app(config=DEFAULT_CONFIG, engine=engine, bus=bus, out_dir=tmp_path), headers=AUTH)


def test_ui_trace_page_is_served_as_html(tmp_path):
    with _client(tmp_path) as client:
        res = client.get("/ui/trace")
        assert res.status_code == 200 and res.headers["content-type"].startswith("text/html")
        assert "<title>SL-RAG Trace</title>" in res.text
        manifest = client.get("/ui/fields.json")
        assert manifest.status_code == 200 and manifest.json()["events"]


def test_api_summary_returns_the_summary_json(tmp_path):
    summary = {"mode": "ours", "metrics": {"recall@5": 1.0, "ttft_p50": 385.0},
               "gates": {"G2_early_retrieval": {"status": "PASS", "metric": "early_retrieval_rate", "condition": ">= 0.80"}},
               "baseline_metrics": {"b1": {"recall@5": 1.0}, "b0": {"recall@5": 0.9}}, "experiments": {}}
    (tmp_path / "a3.json").write_text(json.dumps({"split": "test", "arms": {"off": {"ttft_p50": 490.0}}}), encoding="utf-8")
    with _client(tmp_path, summary) as client:
        res = client.get("/api/summary")
        assert res.status_code == 200 and res.headers["content-type"].startswith("application/json")
        data = res.json()
        assert data["metrics"] == summary["metrics"] and data["gates"] == summary["gates"]
        assert data["experiments"]["A3"]["arms"]["off"]["ttft_p50"] == 490.0  # experiment files next to summary.json


def test_api_summary_is_404_when_absent(tmp_path):
    with _client(tmp_path) as client:
        assert client.get("/api/summary").status_code == 404
