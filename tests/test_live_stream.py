"""Live /ws/stream ingestion runs the streaming controller (WAIT -> PROVISIONAL -> COMMIT / SUPPRESS)."""

import json
from pathlib import Path

from starlette.testclient import TestClient

from slrag.config import DEFAULT_CONFIG, TelemetryConfig
from slrag.corpus.loader import CorpusLoader
from slrag.gateway.session import SessionRegistry
from slrag.pipeline.turn_engine import BatchTurnEngine
from slrag.retrieval.engine import HybridRetrievalEngine
from slrag.server.app import create_app
from slrag.telemetry.bus import TelemetryBus
from slrag.telemetry.metrics import MetricsCollector

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_corpus.json"


def _app(log_file):
    engine = HybridRetrievalEngine(DEFAULT_CONFIG)
    engine.index_documents(CorpusLoader.load_file(CORPUS_PATH))
    bus = TelemetryBus(TelemetryConfig(jsonl_log_path=str(log_file)))
    metrics = MetricsCollector()
    turn_engine = BatchTurnEngine(DEFAULT_CONFIG, engine=engine, bus=bus, metrics=metrics)
    return create_app(config=DEFAULT_CONFIG, engine=engine, bus=bus, registry=SessionRegistry(), metrics=metrics, turn_engine=turn_engine)


def _send_turn(ws, chunks, seq):
    for text in chunks:
        ws.send_text(json.dumps({"event_type": "transcript_chunk", "chunk": text, "seq": seq}))
        seq += 1
    ws.send_text(json.dumps({"event_type": "utterance_end", "seq": seq}))
    out = []
    while True:
        msg = json.loads(ws.receive_text())
        out.append(msg)
        if msg["event_type"] == "turn_summary":
            return out, seq + 1


def test_live_stream_retrieves_early_reuses_evidence_and_suppresses_presentation(tmp_path):
    log_file = tmp_path / "live.jsonl"
    with TestClient(_app(log_file)) as client:
        with client.websocket_connect("/ws/stream?session_id=live_user") as ws:
            first, seq = _send_turn(ws, ["Explain how Raft leader", "election tolerates two failures", "in a five node cluster"], 1)
            second, _ = _send_turn(ws, ["Please repeat your last answer", "in two bullets."], seq)

    events = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    turn1 = first[-1]["turn_id"]
    turn2 = second[-1]["turn_id"]
    assert turn1 != turn2

    def idx(pred):
        return next(i for i, e in enumerate(events) if pred(e))

    # Turn 1: controller decisions per chunk, early retrieval before utterance_end, final path reuses it.
    decisions = [e["decision"] for e in events if e["event_type"] == "controller_decision" and e.get("turn_id") == turn1]
    assert decisions[0] == "WAIT" and "PROVISIONAL" in decisions
    early = idx(lambda e: e["event_type"] == "speculative_retrieval" and e.get("turn_id") == turn1)
    utt_end = idx(lambda e: e["event_type"] == "utterance_end" and e.get("turn_id") == turn1)
    assert early < utt_end
    final_retrievals = [e for e in events if e["event_type"] == "retrieval" and e.get("turn_id") == turn1]
    assert final_retrievals and all(e["mode"] == "cache" for e in final_retrievals)
    assert any(e["event_type"] == "speculation_outcome" and e.get("turn_id") == turn1 and e["outcome"] == "used" for e in events)
    assert first[-1]["verified_count"] >= 1

    # Turn 2: presentation request -> SUPPRESS, answered from the ledger with zero retrievals.
    assert any(e["event_type"] == "controller_decision" and e.get("turn_id") == turn2 and e["decision"] == "SUPPRESS" for e in events)
    assert not [e for e in events if e.get("turn_id") == turn2 and e["event_type"] in ("retrieval", "speculative_retrieval")]
    assert second[-1]["status"] == "presentation"
    answer = next(m for m in second if m["event_type"] == "answer_delta")["text_delta"]
    assert answer.count("- ") == 2


def test_live_second_message_refines_the_session_answer(tmp_path):
    """Phase 7 on /ws/stream: once the session has an answer (answer_version >= 1), a later content
    turn goes through the delta planner and refinement path, as in replay."""
    log_file = tmp_path / "refine.jsonl"
    app = _app(log_file)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream?session_id=refine_user") as ws:
            first, seq = _send_turn(ws, ["How does Okapi BM25 rank", "documents with term frequency?"], 1)
            second, _ = _send_turn(ws, ["Assume the BM25 ranking must", "also handle acronym matching", "and rare entity lookup."], seq)

    events = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    turn1, turn2 = first[-1]["turn_id"], second[-1]["turn_id"]
    assert first[-1]["status"] == "completed"

    def of(etype, turn):
        return [e for e in events if e["event_type"] == etype and e.get("turn_id") == turn]

    # Turn 1 creates answer v1 (initial transition); turn 2 is planned as a delta, not answered from scratch.
    assert [(t["payload"]["from"], t["payload"]["to"]) for t in of("answer_version_transition", turn1)] == [(0, 1)]
    plan = of("plan_completed", turn2)
    assert len(plan) == 1 and plan[0]["payload"]["relation"] == "modifies" and not plan[0]["payload"]["fallback"]
    transition = of("answer_version_transition", turn2)
    assert len(transition) == 1
    payload = transition[0]["payload"]
    assert (payload["from"], payload["to"]) == (1, 2) and payload["unchanged_hashes_ok"] and payload["lineage_ok"]
    assert payload["added"] or payload["revised"]

    # Early retrieval is paused on the refinement turn; only delta retrievals (trigger "refinement") run.
    assert {e["trigger"] for e in of("speculative_retrieval", turn2)} == {"refinement"}
    assert not [e for e in of("retrieval", turn2)]

    # The client gets the claim-level diff and only the new/revised claims; v1 claims are kept byte-identical.
    delta = next(m for m in second if m["event_type"] == "answer_delta")
    assert delta["change_type"] == "refine" and delta["answer_version"] == 2
    kept = {op["claim_id"] for op in delta["ops"] if op["op"] == "keep"}
    assert kept and {m["claim_id"] for m in second if m["event_type"] == "answer_chunk"} == set(payload["added"] + payload["revised"])
    assert second[-1]["status"] == "refined"
    assert all(e.get("cfg_hash") for e in of("answer_version_transition", turn2) + of("plan_completed", turn2))


def test_batch_controller_mode_keeps_live_ingestion_passive(tmp_path):
    import dataclasses

    log_file = tmp_path / "batch.jsonl"
    app = _app(log_file)
    app.state.live.config = dataclasses.replace(DEFAULT_CONFIG, controller=dataclasses.replace(DEFAULT_CONFIG.controller, mode="batch"))
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream?session_id=batch_user") as ws:
            out, _ = _send_turn(ws, ["Explain how Raft leader", "election tolerates two failures"], 1)
    events = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert not [e for e in events if e["event_type"] in ("controller_decision", "speculative_retrieval")]
    assert [e["mode"] for e in events if e["event_type"] == "retrieval"] == ["hybrid"]
    assert out[-1]["event_type"] == "turn_summary"
