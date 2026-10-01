"""Phase 8 commit-stage pre-drafting: launch, hold, timeout, flags, telemetry, live /ws/stream, A3."""

import asyncio
import json
from pathlib import Path
import time

import pytest
from click.testing import CliRunner

from slrag.cli import cli
from slrag.config import DEFAULT_CONFIG
from slrag.eval.clock import ReplayClock
from slrag.eval.runner import InMemoryBus, ReplayRunner
from slrag.replay.baselines import build_index, build_turn_engine, mode_config

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ROOT / "eval" / "scenarios"
# Commits (stable Q_t) early enough that the pre-draft is usable at utterance_end.
QUERY = "Explain how Raft leader election tolerates two failures in a five node cluster"
RECONCILE_FIELDS = {"outcome", "reason", "stands", "refined", "redone", "wasted_tokens", "predraft_tokens", "new_chunk_ids",
                    "dropped_chunk_ids", "discarded_claims", "discarded_predrafts", "predraft_ready_before_end",
                    "predraft_late_ms", "timed_out", "cfg_hash", "turn_id"}


@pytest.fixture(scope="module")
def index():
    return build_index(mode_config("ours").app)


def run(index, query=QUERY, overrides=None, patch=None):
    bus = InMemoryBus()
    engine = build_turn_engine("ours", index, bus, clock=ReplayClock(), overrides=overrides)
    if patch:
        patch(engine)
    result = asyncio.run(engine.execute_turn(query, turn_id="t"))
    return engine, bus.events, result


def of(events, etype):
    return [e for e in events if e.event_type == etype]


def test_config_flag_defaults():
    spec = DEFAULT_CONFIG.speculation
    assert (spec.mode, spec.predraft, spec.provisional) == ("full", True, True) and spec.predraft_on


def test_predraft_runs_at_commit_and_is_held_until_reconcile(index):
    engine, events, result = run(index)
    types = [e.event_type for e in events]
    commit = next(i for i, e in enumerate(events) if e.event_type == "controller_decision" and e.decision == "COMMIT")
    utterance_end = types.index("utterance_final")
    drafts = [i for i, e in enumerate(events) if e.event_type == "draft_completed"]
    assert drafts and all(events[i].speculative for i in drafts)
    # Launched at COMMIT; its results are held: nothing is emitted before utterance_end, and the
    # first answer token follows the reconcile decision (reconcile_completed, with the final claim
    # counts, closes the turn's reconcile afterwards).
    assert commit < utterance_end
    assert "first_token_emission" not in types[:utterance_end]
    reconcile = next(i for i, e in enumerate(events) if e.event_type == "controller_decision" and e.decision == "COMMIT_RECONCILE")
    assert utterance_end < reconcile < types.index("first_token_emission") < types.index("reconcile_completed")
    assert result["predraft_status"] == "committed" and result["reconcile_outcome"] == "stands"


def test_reconcile_event_fields_are_in_the_telemetry(index, tmp_path):
    engine, events, _ = run(index)
    rec = of(events, "reconcile_completed")
    assert len(rec) == 1 and RECONCILE_FIELDS <= set(rec[0].model_dump())
    complete = of(events, "turn_complete")[-1]
    assert complete.reconcile_outcome == rec[0].outcome and complete.wasted_tokens == rec[0].wasted_tokens
    assert complete.predraft_ready_before_end is not None
    decisions = [(e.decision, e.reason) for e in of(events, "controller_decision")]
    assert ("COMMIT_RECONCILE", rec[0].outcome) in decisions

    # Replay telemetry JSONL carries the same fields, and trace coverage requires the reconcile stage.
    out = tmp_path / "t.jsonl"
    ReplayRunner(str(SCENARIOS / "tune.jsonl"), mode="ours", out_path=str(out), category="single-early").run(generate_report=False)
    logged = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    recs = [e for e in logged if e["event_type"] == "reconcile_completed"]
    assert recs and all(RECONCILE_FIELDS <= set(e) for e in recs)
    from slrag.replay.metrics import MetricCalculator
    m = MetricCalculator.calculate_all(logged)
    assert m["trace_coverage"] == 1.0 and m["predraft_turns"] == len(recs)
    stripped = [e for e in logged if e["event_type"] != "reconcile_completed"]
    assert MetricCalculator.calculate_all(stripped)["trace_coverage"] < 1.0


@pytest.mark.parametrize("overrides", [{"speculation.predraft": False}, {"speculation.mode": "retrieval_only"}])
def test_predraft_disabled_launches_no_task(index, overrides, monkeypatch):
    import slrag.answer.predraft as predraft

    launched = []
    monkeypatch.setattr(predraft.Predrafter, "start", lambda self, *a, **k: launched.append(1))
    engine, events, result = run(index, overrides=overrides)
    assert engine.predrafter is None and launched == []
    assert not of(events, "draft_completed") and not of(events, "reconcile_completed")
    assert of(events, "turn_complete")[-1].reconcile_outcome is None
    assert result["output"]  # the normal Phase 5 path answers
    assert any(e.is_early for e in of(events, "speculative_retrieval"))  # early retrieval unchanged


def test_mode_off_disables_early_retrieval_and_predraft(index):
    engine, events, result = run(index, overrides={"speculation.mode": "off"})
    assert engine.predrafter is None
    assert not [e for e in of(events, "speculative_retrieval") if e.is_early]
    assert result["output"]


def test_predraft_timeout_falls_through_to_the_normal_path(index):
    def slow(engine):
        real = engine.predrafter.draft_fn

        def sleepy(text, chunks):
            time.sleep(0.5)
            return real(text, chunks)

        engine.predrafter.draft_fn = sleepy

    engine, events, result = run(index, overrides={"speculation.predraft_timeout_s": 0.05}, patch=slow)
    rec = of(events, "reconcile_completed")[-1]
    assert (rec.outcome, rec.reason, rec.timed_out) == ("redone", "timeout", True)
    assert result["predraft_status"] == "wasted"
    assert result["output"] and of(events, "first_token_emission")  # answered by the normal path
    assert of(events, "verification")


def test_speculative_waste_counts_in_cost(index):
    # A tail that changes the committed quantity: the pre-draft answers a superseded query.
    query = "Explain how Raft leader election tolerates two failures in a five node cluster with seven nodes"
    engine, events, result = run(index, query=query)
    rec = of(events, "reconcile_completed")[-1]
    assert (rec.outcome, rec.reason) == ("redone", "slot_conflict")  # five -> seven after the commit
    complete = of(events, "turn_complete")[-1]
    assert complete.wasted_tokens == rec.wasted_tokens == rec.predraft_tokens > 0
    assert complete.tokens_in + complete.tokens_out > rec.wasted_tokens  # waste + the redraft are both in the cost
    assert result["predraft_status"] == "wasted"
    assert engine.ledger.get_verified_claims()  # the answer comes from the full pipeline on the final text


def test_live_ws_stream_predrafts_and_reconciles(tmp_path):
    from starlette.testclient import TestClient

    from test_live_stream import _app, _send_turn

    log = tmp_path / "live.jsonl"
    with TestClient(_app(log)) as client:
        with client.websocket_connect("/ws/stream?session_id=predraft_user") as ws:
            out, _ = _send_turn(ws, ["Explain how Raft leader election", "and log replication work", "across a majority quorum"], 1)
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    turn = out[-1]["turn_id"]
    drafts = [e for e in events if e["event_type"] == "draft_completed" and e.get("turn_id") == turn]
    rec = [e for e in events if e["event_type"] == "reconcile_completed" and e.get("turn_id") == turn]
    assert drafts and drafts[0]["speculative"]
    assert len(rec) == 1 and RECONCILE_FIELDS <= set(rec[0])
    summary = out[-1]
    assert summary["event_type"] == "turn_summary" and summary["reconcile_outcome"] == rec[0]["outcome"]
    assert summary["wasted_tokens"] == rec[0]["wasted_tokens"]
    if rec[0]["outcome"] == "stands":
        assert summary["verified_count"] == rec[0]["stands"] > 0
        assert summary["tokens_in"] + summary["tokens_out"] == rec[0]["predraft_tokens"]  # no second LLM call


@pytest.mark.integration
def test_a3_three_arms(tmp_path):
    runner = ReplayRunner(str(SCENARIOS), mode="ours", out_path=str(tmp_path / "x.jsonl"), split="tune", category="single-early")
    result = runner.a3_experiment(out_dir=str(tmp_path))
    arms = result["arms"]
    assert list(arms) == ["off", "retrieval_only", "full"]
    for arm in arms.values():
        assert {"ttft_p50", "ttft_p95", "wasted_speculation_rate", "ttft_p50_single-early"} <= set(arm)
    assert arms["off"]["early_retrieval_rate"] == 0.0 and not arms["off"]["predraft_turns"]
    assert not arms["retrieval_only"]["predraft_turns"] and arms["full"]["predraft_turns"] > 0
    assert arms["full"]["ttft_p50"] <= arms["off"]["ttft_p50"]
    assert result["groundedness_parity"] is True
    assert all((tmp_path / f"a3_{arm}.jsonl").exists() for arm in arms)

    (tmp_path / "a3.json").write_text(json.dumps(result), encoding="utf-8")
    report = CliRunner().invoke(cli, ["report", str(tmp_path), "--split", "tune", "--experiment", "A3"])
    assert report.exit_code == 0 and "ttft_p50" in report.output and "retrieval_only" in report.output
