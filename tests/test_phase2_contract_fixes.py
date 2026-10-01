"""Phase 2 fixes: missing schemas, non-blocking telemetry emit, gateway aliases and cumulative merge over /ws/stream."""

import asyncio
import json
from pathlib import Path

import pytest

from slrag.config import DEFAULT_CONFIG, TelemetryConfig
from slrag.contracts.events import (
    AnswerChunk,
    Claim,
    Constraint,
    SessionEnd,
    SubIntent,
    TranscriptChunk,
    TurnOutput,
    compute_text_hash,
)
from slrag.gateway.normalizer import EventNormalizer
from slrag.gateway.session import SessionRegistry
from slrag.server.app import create_app
from slrag.telemetry.bus import TelemetryBus
from slrag.telemetry.metrics import MetricsCollector


def test_missing_schema_models_exist_with_frozen_fields():
    assert SessionEnd(session_id="s1").event_type == "session_end"
    sub = SubIntent(id="q1", text="What is Raft?", status="answerable", sufficiency={"dense_top1": 0.7, "coverage": 0.8})
    assert sub.status == "answerable"
    assert Constraint(constraint_id="k1", text="for 50 people", turn_id="t2", affects=["q1"]).affects == ["q1"]
    chunk = AnswerChunk(turn_id="t1", answer_version=1, claim_id="c1", text="Raft elects a leader.", cites=["DOC002§raft"], first_token=True)
    assert chunk.event_type == "answer_chunk" and chunk.first_token
    out = TurnOutput(sub_queries=["q"], answer="a", citations=["DOC002§raft"], uncertainty=None, meta={"answer_version": 1})
    assert set(out.model_dump()) == {"retrieval_events", "sub_queries", "answer", "citations", "uncertainty", "meta"}
    with pytest.raises(Exception):
        SubIntent(id="q1", text="x", status="unknown-status")


def test_claim_text_hash_is_deterministic_and_serialized():
    a = Claim(claim_id="c1", text="Raft tolerates two failures.", doc_ids=["B§2", "A§1"])
    b = Claim(claim_id="c9", text="Raft tolerates two failures.", doc_ids=["A§1", "B§2"])
    assert a.text_hash == b.text_hash == compute_text_hash("Raft tolerates two failures.", ["A§1", "B§2"])
    assert a.text_hash != Claim(text="Raft tolerates three failures.", doc_ids=["A§1", "B§2"]).text_hash
    assert a.model_dump()["text_hash"] == a.text_hash


@pytest.mark.anyio
async def test_emit_is_non_blocking_and_lossless_when_queue_is_full(tmp_path):
    log_file = tmp_path / "overflow.jsonl"
    bus = TelemetryBus(TelemetryConfig(jsonl_log_path=str(log_file), buffer_size=10))
    bus.start()
    events = [TranscriptChunk(seq=i, text=f"chunk {i}") for i in range(200)]

    # 200 emits into a queue of 10 without yielding to the writer: must neither block nor raise.
    for evt in events:
        bus.emit_nowait(evt)
    assert bus.overflow_count > 0

    await bus.close()
    lines = [json.loads(line)["event_id"] for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert lines == [e.event_id for e in events]


@pytest.mark.anyio
async def test_async_emit_never_waits_on_a_full_queue(tmp_path):
    bus = TelemetryBus(TelemetryConfig(jsonl_log_path=str(tmp_path / "t.jsonl"), buffer_size=1))
    bus.start()
    await asyncio.wait_for(asyncio.gather(*(bus.emit(TranscriptChunk(seq=i, text="x")) for i in range(50))), timeout=0.5)
    await bus.close()
    assert bus.total_written == 50


def test_gateway_aliases_chunk_text_and_timestamps():
    n = EventNormalizer()
    assert n.normalize_dict_keys({"chunk": "hello"})["text"] == "hello"
    assert n.normalize_dict_keys({"text": "a", "chunk": "b"})["text"] == "a"
    for key in ("timestamp_s", "timestamp", "t"):
        assert n.normalize_dict_keys({key: "1.25"})["ts_s"] == 1.25
    assert n.normalize_dict_keys({"ts_s": 2.0, "t": 9})["ts_s"] == 2.0


def test_cumulative_and_delta_merge_modes():
    n = EventNormalizer()
    assert n.merge("I need to plan")[2] == "delta"  # first chunk starts the buffer
    _, buf, mode = n.merge("I need to plan a workshop in")
    assert (buf, mode) == ("I need to plan a workshop in", "cumulative")
    _, buf, mode = n.merge("...Pune for 30")
    assert mode == "delta" and buf.endswith("...Pune for 30")
    _, buf, mode = n.merge("and I need…")
    assert mode == "delta" and buf.endswith("and I need…")
    # Normalised comparison: case and whitespace differences still count as cumulative.
    m = EventNormalizer()
    m.merge("hello world")
    assert m.merge("Hello   world again")[2] == "cumulative"


def test_ws_stream_merges_cumulative_chunks_into_utterance(tmp_path):
    from starlette.testclient import TestClient

    log_file = tmp_path / "ws.jsonl"
    bus = TelemetryBus(TelemetryConfig(jsonl_log_path=str(log_file)))
    app = create_app(config=DEFAULT_CONFIG, bus=bus, registry=SessionRegistry(), metrics=MetricsCollector())

    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream?session_id=merge_user") as ws:
            ws.send_text(json.dumps({"event_type": "transcript_chunk", "chunk": "how does raft", "seq": 1, "t": 0.0}))
            ws.send_text(json.dumps({"event_type": "transcript_chunk", "chunk": "how does raft elect a leader", "seq": 2, "t": 0.8}))
            ws.send_text(json.dumps({"event_type": "utterance_end", "seq": 3, "t": 1.2}))
            ws.send_text(json.dumps({"event_type": "session_end", "seq": 4}))

    events = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    chunks = [e for e in events if e["event_type"] == "transcript_chunk"]
    assert [c["merge_mode"] for c in chunks] == ["delta", "cumulative"]
    assert chunks[-1]["buffer_text"] == "how does raft elect a leader"
    assert chunks[-1]["ts_s"] == 0.8
    utt = next(e for e in events if e["event_type"] == "utterance_end")
    assert utt["final_text"] == "how does raft elect a leader"
    assert any(e["event_type"] == "session_end" for e in events)
