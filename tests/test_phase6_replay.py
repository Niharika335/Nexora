"""Tests for Phase 6 Deterministic Replay, Baselines, and Gates."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from slrag.config import SlragConfig
from slrag.contracts.events import ReconciliationEvent
from slrag.eval.clock import ReplayClock
from slrag.eval.gates import GateEvaluator
from slrag.eval.runner import ReplayRunner
from slrag.eval.scenarios import generate_75_scenarios, make_stratified_split, write_split_files
from slrag.pipeline.turn_engine import TurnEngine
from slrag.telemetry.metrics import MetricCalculator


class MockBus:
    def __init__(self):
        self.events = []
    def publish(self, event):
        self.events.append(event)


class MockChunk:
    def __init__(self, cid):
        self.id = cid


class MockRetrieval:
    def __init__(self):
        self.call_count = 0
    def retrieve(self, query):
        self.call_count += 1
        return [MockChunk("doc-nexora-arch-c0")]


class MockLLM:
    def generate(self, prompt, temperature=0.0, seed=13):
        return "Generated answer"


class MockDrafter:
    def draft(self, query, chunks, temperature=0.0, seed=13):
        return f"Draft for {query}"


class MockSufficiency:
    def evaluate(self, query, chunks):
        class Res:
            sufficient = True
            score = 0.95
            reason = ""
        return Res()


def test_generate_exact_75_scenarios():
    scenarios = generate_75_scenarios()
    assert len(scenarios) == 75

    categories = {}
    for s in scenarios:
        categories[s["category"]] = categories.get(s["category"], 0) + 1

    assert categories["compound"] == 30
    assert categories["single-early"] == 15
    assert categories["presentation"] == 15
    assert categories["unanswerable"] == 15


def test_stratified_split_counts_seed_13():
    scenarios = generate_75_scenarios()
    tune, test = make_stratified_split(scenarios, seed=13)

    assert len(tune) == 36
    assert len(test) == 39

    def cat_count(s_list):
        c = {}
        for x in s_list:
            c[x["category"]] = c.get(x["category"], 0) + 1
        return c

    tune_c = cat_count(tune)
    test_c = cat_count(test)

    assert tune_c["compound"] == 15
    assert test_c["compound"] == 15
    assert tune_c["single-early"] == 7
    assert test_c["single-early"] == 8
    assert tune_c["presentation"] == 7
    assert test_c["presentation"] == 8
    assert tune_c["unanswerable"] == 7
    assert test_c["unanswerable"] == 8


def test_no_test_leakage_into_calibration():
    from slrag.eval.calibrate import CalibrationSweep
    with pytest.raises(ValueError, match="Test scenarios must NEVER be used for calibration"):
        CalibrationSweep.run_sweep("eval/scenarios/test.jsonl")


def test_semantic_gold_chunks_exist_in_corpus():
    from slrag.config import DEFAULT_CONFIG
    from slrag.replay.baselines import build_index

    scenarios = generate_75_scenarios()
    valid_ids = set(build_index(DEFAULT_CONFIG).chunks_map)  # chunk ids actually indexed by the replay harness

    for sc in scenarios:
        if sc["is_unanswerable"]:
            for s in sc["sub_intents"]:
                assert s["gold_chunk_ids"] == []
        else:
            for s in sc["sub_intents"]:
                assert len(s["gold_chunk_ids"]) > 0
                for cid in s["gold_chunk_ids"]:
                    assert cid in valid_ids


@pytest.mark.asyncio
async def test_baseline_restart_on_late_detail():
    bus = MockBus()
    cfg_b0 = SlragConfig(
        dense_weight=1.0,
        bm25_weight=0.0,
        enable_verifier=False,
        enable_cascade=False,
        enable_multi_intent=False,
        restart_on_late_detail=True
    )
    retrieval = MockRetrieval()
    engine = TurnEngine(
        config=cfg_b0,
        retrieval_engine=retrieval,
        llm=MockLLM(),
        verifier=None,
        sufficiency=MockSufficiency(),
        ledger=None,
        drafter=MockDrafter(),
        bus=bus
    )

    res = await engine.execute_turn(
        query="Initial query",
        turn_id="t_late",
        late_detail="with late arriving parameter"
    )

    assert retrieval.call_count == 2
    rec_events = [e for e in bus.events if isinstance(e, ReconciliationEvent)]
    assert len(rec_events) == 1
    assert rec_events[0].reconciliation_type == "restart"


def test_gate_evaluation_logic():
    ours = {"recall@10": 1.0, "groundedness": 0.95, "early_retrieval_rate": 0.82, "hallucinated_id_rate": 0.0, "trace_coverage": 1.0}
    b1 = {"recall@10": 0.90, "groundedness": 0.95}
    b0 = {"recall@10": 0.80, "groundedness": None}

    gates = GateEvaluator.evaluate_gates(ours, b1, b0)
    assert gates["G1_recall_improvement"]["status"] == "PASS"
    assert gates["G2_early_retrieval"]["status"] == "PASS"
    assert gates["G3_groundedness_preservation"]["status"] == "PASS"
    assert gates["G4_grounding"]["status"] == "PASS"
    assert gates["G6_trace_coverage"]["status"] == "PASS"

    failing = dict(ours, early_retrieval_rate=0.79, hallucinated_id_rate=0.01, trace_coverage=0.99)
    gates = GateEvaluator.evaluate_gates(failing, b1, b0)
    assert gates["G2_early_retrieval"]["status"] == "FAIL"
    assert gates["G4_grounding"]["status"] == "FAIL"
    assert gates["G6_trace_coverage"]["status"] == "FAIL"
    assert GateEvaluator.evaluate_gates(dict(ours, groundedness=0.84), b1, b0)["G4_grounding"]["status"] == "FAIL"
    # A missing metric fails its gate instead of passing silently.
    assert GateEvaluator.evaluate_gates({"recall@10": 1.0}, b1, b0)["G6_trace_coverage"]["status"] == "FAIL"


def test_replay_runner_execution_and_reports(tmp_path):
    write_split_files("eval/scenarios")
    out_file = tmp_path / "ours.jsonl"
    runner = ReplayRunner(
        scenarios_path="eval/scenarios/test.jsonl",
        mode="ours",
        out_path=str(out_file),
        playback="virtual"
    )
    res = runner.run(generate_report=True)

    assert out_file.exists()
    summary_file = tmp_path / "summary.json"
    report_file = tmp_path / "report.md"
    assert summary_file.exists()
    assert report_file.exists()

    with open(summary_file, "r") as f:
        summary_data = json.load(f)
    assert summary_data["scenario_counts"]["total"] == 75
    assert summary_data["scenario_counts"]["test"] == 39


def test_clock_import_and_tick():
    from slrag.eval.clock import ReplayClock
    clock = ReplayClock(start_time=100.0, tick_interval=0.05)
    assert clock.time() == 100.0
    assert clock.advance() == 100.05


def test_genuine_b0_b1_execution_modes(tmp_path):
    write_split_files("eval/scenarios")
    runner_b0 = ReplayRunner("eval/scenarios/test.jsonl", mode="b0", out_path=str(tmp_path / "b0.jsonl"))
    res_b0 = runner_b0.run(generate_report=False)
    assert res_b0["metrics"]["groundedness"] is None

    runner_b1 = ReplayRunner("eval/scenarios/test.jsonl", mode="b1", out_path=str(tmp_path / "b1.jsonl"))
    res_b1 = runner_b1.run(generate_report=False)
    assert res_b1["metrics"]["groundedness"] is not None


def test_calibration_sweep_tune_only():
    from slrag.eval.calibrate import CalibrationSweep
    write_split_files("eval/scenarios")
    cfg = CalibrationSweep.run_sweep("eval/scenarios/tune.jsonl")
    assert cfg["seed"] == 13
    assert "cascade_confidence_threshold" in cfg
