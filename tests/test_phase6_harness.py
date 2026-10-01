"""Phase 6: metric set, split hygiene, scenario validation, determinism, baseline sanity, CLI."""

import json
from pathlib import Path
import subprocess
import sys

import pytest
from click.testing import CliRunner

from slrag.cli import cli
from slrag.eval.runner import ReplayRunner, load_scenarios
from slrag.replay.metrics import REQUIRED_METRICS, MetricCalculator

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ROOT / "eval" / "scenarios"


@pytest.fixture(scope="module")
def ours_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("ours") / "telemetry.jsonl"
    result = ReplayRunner(str(SCENARIOS / "test.jsonl"), mode="ours", out_path=str(out)).run(generate_report=False)
    return result, [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


def test_all_required_metrics_are_computed(ours_run):
    metrics = ours_run[0]["metrics"]
    missing = [m for m in REQUIRED_METRICS if metrics.get(m) is None]
    assert missing == []
    assert metrics["recall@10"] > 0.5 and metrics["early_retrieval_rate"] > 0.0


def test_every_event_carries_cfg_hash(ours_run):
    events = ours_run[1]
    assert events and all(e.get("cfg_hash") for e in events)


def test_metric_fixtures_for_new_metrics():
    events = [
        {"event_type": "turn_start", "turn_id": "a", "timestamp": 10.0},
        {"event_type": "utterance_final", "turn_id": "a", "timestamp": 12.0},
        {"event_type": "first_token_emission", "turn_id": "a", "timestamp": 12.25},
        {"event_type": "turn_complete", "turn_id": "a", "est_cost_usd": 0.002},
        {"event_type": "turn_complete", "turn_id": "b", "est_cost_usd": 0.004},
        {"event_type": "multi_intent_resolution", "turn_id": "a", "total_sub_intents": 3, "resolved_count": 2, "gold_sub_intents": 2},
        {"event_type": "multi_intent_resolution", "turn_id": "b", "total_sub_intents": 1, "resolved_count": 1, "gold_sub_intents": 2},
        {"event_type": "sub_intent_retrieval", "turn_id": "a", "retrieved_chunk_ids": ["x", "g1"], "expected_chunk_ids": ["g1", "g2"]},
        {"event_type": "sub_intent_retrieval", "turn_id": "a", "retrieved_chunk_ids": ["g2"], "expected_chunk_ids": ["g1", "g2"]},
        {"event_type": "speculative_cache", "action": "set"},
        {"event_type": "speculative_cache", "action": "hit"},
        {"event_type": "speculative_cache", "action": "miss"},
        {"event_type": "speculative_cache", "action": "miss"},
    ]
    m = MetricCalculator.calculate_all(events)
    assert m["ttft_p50"] == pytest.approx(250.0)  # measured from utterance_end, not turn_start
    assert m["cost_per_turn"] == pytest.approx(0.003)
    assert m["subintent_recall"] == pytest.approx(3 / 4)
    assert m["over_fragmentation"] == pytest.approx(0.5)
    assert m["recall@5"] == 1.0  # recall is per turn over the union of its sub-queries
    assert m["cache_hit_rate"] == pytest.approx(1 / 3)  # "set" is not a lookup


def test_split_manifest_hygiene():
    split = json.loads((ROOT / "eval" / "split.json").read_text(encoding="utf-8"))
    tune, test = set(split["tune"]), set(split["test"])
    assert split["seed"] == 13 and not tune & test
    assert tune == {s["scenario_id"] for s in load_scenarios(SCENARIOS / "tune.jsonl")}
    assert test == {s["scenario_id"] for s in load_scenarios(SCENARIOS / "test.jsonl")}
    assert len(tune) + len(test) >= 75


def test_validate_scenarios_script_passes():
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_scenarios.py"), str(SCENARIOS)],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip().endswith("OK")


def test_validate_scenarios_script_rejects_unknown_gold_id(tmp_path):
    scen = tmp_path / "scenarios"
    scen.mkdir()
    bad = {"scenario_id": "x1", "category": "single-early", "query": "q", "is_unanswerable": False,
           "sub_intents": [{"id": "sub_1", "query": "q", "gold_chunk_ids": ["doc-nexora-arch-c0"]}]}
    (scen / "test.jsonl").write_text(json.dumps(bad) + "\n", encoding="utf-8")
    (tmp_path / "split.json").write_text(json.dumps({"seed": 13, "tune": [], "test": ["x1"]}), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_scenarios.py"), str(scen)],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 1 and "not in the indexed corpus" in proc.stdout


def test_replay_is_deterministic(tmp_path):
    def run(i):
        out = tmp_path / f"r{i}.jsonl"
        metrics = ReplayRunner(str(SCENARIOS / "tune.jsonl"), mode="ours", out_path=str(out)).run(generate_report=False)["metrics"]
        claims = [json.loads(l)["output_text"] for l in out.read_text(encoding="utf-8").splitlines() if '"turn_complete"' in l]
        return metrics, claims

    (m1, c1), (m2, c2) = run(1), run(2)
    assert c1 == c2 and m1 == m2


def test_b0_makes_one_retrieval_per_content_turn(tmp_path):
    out = tmp_path / "b0.jsonl"
    ReplayRunner(str(SCENARIOS / "test.jsonl"), mode="b0", out_path=str(out)).run(generate_report=False)
    events = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    completes = [e for e in events if e["event_type"] == "turn_complete"]
    assert len(completes) == len(load_scenarios(SCENARIOS / "test.jsonl"))
    assert {e["retrieval_calls"] for e in completes} == {1}
    assert not [e for e in events if e["event_type"] == "speculative_retrieval" and e["is_early"]]
    assert not [e for e in events if e["event_type"] == "verification"]


def test_cli_replay_dry_run_and_coverage(tmp_path):
    runner = CliRunner()
    dry = runner.invoke(cli, ["replay", str(SCENARIOS / "test.jsonl"), "--mode", "b1", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert json.loads(dry.output)["scenarios"] == 39

    out = tmp_path / "t.jsonl"
    ReplayRunner(str(SCENARIOS / "test.jsonl"), mode="b1", out_path=str(out)).run(generate_report=False)
    cov = runner.invoke(cli, ["coverage", str(out)])
    assert cov.exit_code == 0 and "trace_coverage=" in cov.output
