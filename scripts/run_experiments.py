#!/usr/bin/env python3
"""Run the A1-A5 ablations and write out/final_experiments.json (served by GET /api/experiments).

A1: retrieval  hybrid (BM25 + dense) vs dense-only vs BM25-only.
A2: controller eager vs rules_only vs cascade vs llm_only (controller.mode).
A3: speculation off vs retrieval_only vs full (TTFT, cost, waste, groundedness).
A4: late-detail refinement (delta) vs restarting the answer (retrieval calls and tokens on turn 2).
A5: fail-closed verifier off vs on. With the verifier off nothing publishes a verification score, so
    both arms are scored post hoc: every emitted claim is re-checked with the same verifier rules
    against the chunks it cites ("posthoc_support_rate").

Every arm runs the "ours" pipeline with one setting changed; all other settings are the frozen config.yaml.
Timing is the replay clock's latency model (virtual time), not hardware latency.

Usage: python scripts/run_experiments.py [--split test|tune] [--out out/final_experiments.json]
The test split is the evaluation split: run it once, after calibration on tune.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from slrag.config import DEFAULT_CONFIG  # noqa: E402
from slrag.contracts.events import Claim  # noqa: E402
from slrag.eval.runner import ReplayRunner  # noqa: E402
from slrag.pipeline.verifier import FailClosedVerifier  # noqa: E402
from slrag.replay.metrics import MetricCalculator  # noqa: E402

LABELS = {
    "A1": "Retrieval: hybrid (BM25 + dense) vs dense-only vs BM25-only",
    "A2": "Controller: eager vs rules_only vs cascade vs llm_only",
    "A3": "Speculation: off / early retrieval only / early retrieval + pre-drafting",
    "A4": "Late-detail refinement (delta) vs restarting the answer",
    "A5": "Fail-closed verifier off vs on",
}
A1_ARMS = {"hybrid": {}, "dense": {"bm25_weight": 0.0, "dense_weight": 1.0}, "bm25": {"bm25_weight": 1.0, "dense_weight": 0.0}}
A1_METRICS = ("recall@5", "recall@10", "sub_intent_recall", "groundedness", "uncertainty_precision", "uncertainty_recall")
A2_ARMS = ("eager", "rules_only", "cascade", "llm_only")
A2_METRICS = ("early_retrieval_rate", "false_trigger_rate", "wasted_speculation_rate", "cache_hit_rate", "ttft_p50",
              "ttft_p95", "cost_per_turn", "recall@10", "groundedness")
A5_ARMS = {"on": {"enable_verifier": True}, "off": {"enable_verifier": False}}
A5_METRICS = ("groundedness", "hallucinated_id_rate", "ttft_p50", "cost_per_turn", "recall@10")


def run_arm(args: argparse.Namespace, overrides: Dict[str, Any]) -> tuple[ReplayRunner, List[Dict[str, Any]]]:
    runner = ReplayRunner(args.scenarios, mode="ours", out_path=str(Path(args.out).parent / "ablation.jsonl"),
                          split=args.split, config_overrides=overrides)
    return runner, runner._execute_scenarios("ours", runner.load())


def pick(metrics: Dict[str, Any], keys: tuple) -> Dict[str, Any]:
    return {k: metrics.get(k) for k in keys}


def posthoc_support(events: List[Dict[str, Any]], chunks_map: Dict[str, Any]) -> Dict[str, Any]:
    """Re-check every emitted claim with the verifier rules against the chunks it cites."""
    verifier = FailClosedVerifier(DEFAULT_CONFIG.verifier)
    total = passed = 0
    for e in events:
        if e.get("event_type") != "answer_chunk":
            continue
        cites = [c for c in e.get("cites") or [] if c]
        text = e.get("text") or ""
        total += 1
        ok, _, _ = verifier.verify_claim(Claim(text=text, doc_ids=cites), chunks_map, set(chunks_map), [])
        passed += ok
    return {"emitted_claims": total, "posthoc_supported": passed, "posthoc_support_rate": passed / total if total else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["test", "tune"], default="test")
    parser.add_argument("--scenarios", default=str(ROOT / "eval" / "scenarios"))
    parser.add_argument("--out", default=str(ROOT / "out" / "final_experiments.json"))
    args = parser.parse_args()
    out = Path(args.out)

    a1 = {}
    for arm, overrides in A1_ARMS.items():
        _, events = run_arm(args, overrides)
        a1[arm] = pick(MetricCalculator.calculate_all(events), A1_METRICS)
        print(f"A1 {arm}: recall@10={a1[arm]['recall@10']:.3f}")

    a2 = {}
    for arm in A2_ARMS:
        _, events = run_arm(args, {"controller.mode": arm})
        a2[arm] = pick(MetricCalculator.calculate_all(events), A2_METRICS)
        print(f"A2 {arm}: early={a2[arm]['early_retrieval_rate']:.3f} false_trigger={a2[arm]['false_trigger_rate']:.3f}")

    a3 = ReplayRunner(args.scenarios, mode="ours", out_path=str(out.parent / "a3.jsonl"), split=args.split).a3_experiment(
        out_dir=str(out.parent))
    print("A3 done")

    runner = ReplayRunner(args.scenarios, mode="ours", out_path=str(out.parent / "a4.jsonl"), split=args.split, category="late_detail")
    scenarios = runner.load()
    a4 = runner.a4_experiment(scenarios, runner._execute_scenarios("ours", scenarios))
    print("A4 done")

    a5 = {}
    for arm, overrides in A5_ARMS.items():
        runner, events = run_arm(args, overrides)
        row = pick(MetricCalculator.calculate_all(events), A5_METRICS)
        if arm == "off":
            row["groundedness"] = None  # no verification events are published with the verifier off
        row.update(posthoc_support(events, runner._index_for("ours").chunks_map))
        a5[arm] = row
        print(f"A5 verifier {arm}: posthoc_support_rate={row['posthoc_support_rate']}")

    result = {
        "split": args.split,
        "cfg_hash": DEFAULT_CONFIG.cfg_hash,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timing": "virtual replay clock (LatencyModel), not wall-clock hardware latency",
        "experiments": {
            "A1": {"label": LABELS["A1"], "arms": a1},
            "A2": {"label": LABELS["A2"], "arms": a2},
            "A3": {"label": LABELS["A3"], **a3},
            "A4": {"label": LABELS["A4"], **(a4 or {})},
            "A5": {"label": LABELS["A5"], "arms": a5,
                   "note": "posthoc_support_rate re-checks every emitted claim with the verifier rules; "
                           "groundedness is only published when the verifier runs"},
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out} (split={args.split})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
