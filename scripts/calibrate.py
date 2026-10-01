#!/usr/bin/env python3
"""Calibrate the cascade controller (new_info_cos x stable_cos) on the TUNE split only.

Selection rule (fixed before any run):
  1. prefer operating points with false_trigger_rate == 0.0;
  2. among them, the highest early_retrieval_rate (the G2 target is >= 0.80);
  3. if no point has zero false triggers: the lowest false_trigger_rate, then the highest early rate;
  ties: lower wasted_speculation_rate, then the point closest to the current config.yaml values.
The test split is refused: calibrate here, then measure the test split once.

Usage: python scripts/calibrate.py --split tune [--new-info 0.74,0.78,0.82,0.86] [--stable 0.78,0.82,0.85,0.88]
                                   [--scenarios eval/scenarios] [--out out/calibration_tune.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from slrag.config import DEFAULT_CONFIG  # noqa: E402
from slrag.eval.runner import ReplayRunner  # noqa: E402

NEW_INFO_COS = [0.74, 0.78, 0.82, 0.86]
STABLE_COS = [0.78, 0.82, 0.85, 0.88]
G2_TARGET = 0.80


def sweep(scenario_dir: Path, new_info_grid: List[float], stable_grid: List[float]) -> List[Dict[str, Any]]:
    rows = []
    for new_info in new_info_grid:
        for stable in stable_grid:
            overrides = {"controller.new_info_cos": new_info, "controller.stable_cos": stable}
            runner = ReplayRunner(str(scenario_dir), mode="ours", out_path="/dev/null", split="tune", config_overrides=overrides)
            m = runner.run_metrics()
            rows.append({
                "new_info_cos": new_info, "stable_cos": stable,
                "early_retrieval_rate": m["early_retrieval_rate"], "false_trigger_rate": m["false_trigger_rate"],
                "wasted_speculation_rate": m["wasted_speculation_rate"], "recall@10": m["recall@10"],
                "ttft_p50": m["ttft_p50"],
            })
    return rows


def pick(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    current = (DEFAULT_CONFIG.controller.new_info_cos, DEFAULT_CONFIG.controller.stable_cos)
    return max(rows, key=lambda r: (
        r["false_trigger_rate"] == 0.0,
        -round(r["false_trigger_rate"], 9),
        round(r["early_retrieval_rate"], 9),
        -round(r["wasted_speculation_rate"], 9),
        -(abs(r["new_info_cos"] - current[0]) + abs(r["stable_cos"] - current[1])),
    ))


def floats(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="tune")
    parser.add_argument("--new-info", type=floats, default=NEW_INFO_COS, help="controller.new_info_cos grid (comma-separated)")
    parser.add_argument("--stable", type=floats, default=STABLE_COS, help="controller.stable_cos grid (comma-separated)")
    parser.add_argument("--scenarios", default=str(ROOT / "eval" / "scenarios"))
    parser.add_argument("--out", default=str(ROOT / "out" / "calibration_tune.json"))
    args = parser.parse_args()
    if args.split != "tune":
        parser.error("calibration runs on the tune split only; the test split is measured once afterwards")

    rows = sweep(Path(args.scenarios), args.new_info, args.stable)
    best = pick(rows)
    meets = best["false_trigger_rate"] == 0.0 and best["early_retrieval_rate"] >= G2_TARGET
    print("Tune-split sweep (rule: false_trigger_rate == 0, then max early_retrieval_rate; G2 target >= 0.80)")
    print(f"{'new_info_cos':>12} {'stable_cos':>10} {'early':>7} {'false_trig':>10} {'wasted':>7} {'recall@10':>9} {'ttft_p50':>8}")
    for r in rows:
        mark = "  <== chosen" if r is best else ""
        print(f"{r['new_info_cos']:>12.2f} {r['stable_cos']:>10.2f} {r['early_retrieval_rate']:>7.4f} {r['false_trigger_rate']:>10.4f} "
              f"{r['wasted_speculation_rate']:>7.4f} {r['recall@10']:>9.4f} {r['ttft_p50']:>8.1f}{mark}")
    gap = "" if meets else f" -- does NOT reach the G2 target: gap {G2_TARGET - best['early_retrieval_rate']:+.4f}"
    print(f"chosen: new_info_cos={best['new_info_cos']} stable_cos={best['stable_cos']} "
          f"early={best['early_retrieval_rate']:.4f} false_trigger={best['false_trigger_rate']:.4f}{gap}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": "tune", "grid": {"new_info_cos": args.new_info, "stable_cos": args.stable},
                               "g2_target": G2_TARGET, "chosen": best, "meets_target": meets, "sweep": rows}, indent=2),
                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
