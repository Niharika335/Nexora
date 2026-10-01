"""Phase 9 UI contract: the trace page reads only fields listed in slrag/ui/fields.json, and every
required field in the manifest is actually emitted by the engine (the UI shows only what is logged).
Unit test: no browser and no server process (the live gateway runs in-process via TestClient)."""

import asyncio
import json
from pathlib import Path
import re

import pytest

from slrag.eval.clock import ReplayClock
from slrag.eval.runner import InMemoryBus, ReplayRunner
from slrag.replay.baselines import build_index, build_turn_engine, mode_config

ROOT = Path(__file__).resolve().parent.parent
HTML = (ROOT / "slrag" / "ui" / "trace.html").read_text(encoding="utf-8")
MANIFEST = json.loads((ROOT / "slrag" / "ui" / "fields.json").read_text(encoding="utf-8"))
SCRIPT = "\n".join(re.findall(r"<script>(.*?)</script>", HTML, re.S))


def allowed_paths():
    """Every manifest path, plus its tail after any list / map prefix (val(item, "chunk_id") inside evidence)."""
    paths = set(MANIFEST["common"])
    for spec in MANIFEST["events"].values():
        paths.update(spec["required"])
        paths.update(spec["optional"])
    paths.update(MANIFEST["summary"])
    for fields in MANIFEST["api"].values():
        paths.update(fields)
    for p in list(paths):
        parts = p.split(".")
        for i in range(1, len(parts)):
            paths.add(".".join(parts[i:]))
            if parts[i - 1] == "*":
                paths.add(".".join(parts[i:]))
    return paths


def test_every_val_path_in_the_page_is_in_the_manifest():
    used = set(re.findall(r'\bval\(\s*[A-Za-z_$][\w$]*\s*,\s*"([^"]+)"\s*\)', SCRIPT))
    assert used, "the page should read telemetry through val()"
    missing = sorted(used - allowed_paths())
    assert missing == [], f"trace.html reads fields not declared in fields.json: {missing}"


def test_no_direct_property_access_on_telemetry_objects():
    # `.property` and ["key"] reads on event objects must go through val() so they are checked above.
    direct = re.findall(r"\bev\.([A-Za-z_$][\w$]*)", SCRIPT)
    assert direct == [], f"direct event property reads: {direct}"
    keyed = re.findall(r'(?<=[\w$)\]])\[\s*"([^"]+)"\s*\]', SCRIPT)
    assert sorted(set(keyed) - allowed_paths()) == [], f'["key"] reads not in fields.json: {keyed}'


def test_every_event_type_the_page_renders_is_in_the_manifest():
    types = set(re.findall(r'ofType\([^,]+,\s*"([a-z_]+)"\)', SCRIPT)) | set(re.findall(r'type\(ev\)\s*===\s*"([a-z_]+)"', SCRIPT))
    assert types and sorted(types - set(MANIFEST["events"])) == []


def test_page_is_self_contained():
    assert not re.search(r'<script[^>]+src=', HTML) and not re.search(r'<link[^>]+href=', HTML)
    assert not re.search(r'https?://', HTML), "no external requests: the page must work offline"


# -- dynamic: required manifest fields are emitted ------------------------------------------------


def has_path(obj, parts):
    if not parts:
        return True
    if isinstance(obj, list):
        return any(has_path(item, parts) for item in obj)
    if not isinstance(obj, dict):
        return False
    if parts[0] == "*":
        return any(has_path(v, parts[1:]) for v in obj.values())
    return parts[0] in obj and has_path(obj[parts[0]], parts[1:])


@pytest.fixture(scope="module")
def emitted(tmp_path_factory):
    """Telemetry from replay (compound, late-detail refinement, unanswerable, pre-draft, presentation) and a live
    /ws/stream session (gateway events)."""
    events = []
    out = tmp_path_factory.mktemp("ui") / "t.jsonl"
    for split, category in (("test", "compound"), ("test", "late_detail"), ("test", "unanswerable")):
        runner = ReplayRunner(str(ROOT / "eval" / "scenarios"), mode="ours", out_path=str(out), split=split, category=category)
        runner.run(generate_report=False)
        events += [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]

    index = build_index(mode_config("ours").app)
    bus, clock = InMemoryBus(), ReplayClock()
    engine = build_turn_engine("ours", index, bus, clock=clock)
    asyncio.run(engine.execute_turn("Explain how Raft leader election tolerates two failures in a five node cluster", turn_id="p1"))
    clock.advance(2.0)
    asyncio.run(engine.execute_turn("Please repeat your last answer in two bullets.", turn_id="p2"))
    events += [e.model_dump(mode="json") for e in bus.events]

    from starlette.testclient import TestClient
    from test_live_stream import _app, _send_turn

    log = tmp_path_factory.mktemp("live") / "live.jsonl"
    with TestClient(_app(log)) as client:
        with client.websocket_connect("/ws/stream?session_id=contract") as ws:
            _send_turn(ws, ["Explain how Raft leader election", "and log replication work", "across a majority quorum"], 1)
    events += [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    return events


def test_every_required_manifest_field_is_emitted(emitted):
    by_type = {}
    for e in emitted:
        by_type.setdefault(e.get("event_type"), []).append(e)
    problems = []
    for etype, spec in MANIFEST["events"].items():
        if etype not in by_type:
            problems.append(f"{etype}: never emitted")
            continue
        for path in spec["required"] + MANIFEST["common"]:
            if path == "session_id" and etype not in by_type:
                continue
            if not any(has_path(e, path.split(".")) for e in by_type[etype]):
                problems.append(f"{etype}.{path}")
    assert problems == [], f"fields the UI reads but the engine never emits: {problems}"


def test_optional_fields_are_emitted_somewhere(emitted):
    # Optional = not on every event of the type, but the engine must still be able to produce it.
    by_type = {}
    for e in emitted:
        by_type.setdefault(e.get("event_type"), []).append(e)
    for etype, spec in MANIFEST["events"].items():
        for path in spec["optional"]:
            if path == "ops.prev_text_hash":
                continue  # emitted only by a revise op; covered by tests/test_delta_apply.py
            assert any(has_path(e, path.split(".")) for e in by_type.get(etype, [])), f"{etype}.{path}"
