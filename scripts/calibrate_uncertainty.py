#!/usr/bin/env python3
"""Calibrate the uncertainty flag on the tune split: grid over the sufficiency gate's dense_top1 threshold
(the cosine-based score; below it a sub-intent is suppressed and flagged) and the upper edge of the
coverage band that flags an answered sub-intent as uncertain. Objective: F1 of the uncertainty flag
against the gold unanswerable label; ties go to higher sub-intent recall, then the current config value.

Usage: python scripts/calibrate_uncertainty.py [--split tune] [--out out/calibration_uncertainty.json]
Never run it on the test split: the chosen values are written into config.yaml by hand and frozen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from slrag.config import DEFAULT_CONFIG  # noqa: E402
from slrag.eval.runner import ReplayRunner  # noqa: E402
from slrag.replay.metrics import MetricCalculator  # noqa: E402

DENSE_TOP1 = [0.55, 0.60, 0.65, 0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85]
UNCERTAIN_HIGH = [0.50, 0.55, 0.60, 0.65, 0.70]  # 0.50 = the band is empty (no answered sub-intent is flagged)


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["tune"], default="tune")
    parser.add_argument("--scenarios", default=str(ROOT / "eval" / "scenarios"))
    parser.add_argument("--out", default=str(ROOT / "out" / "calibration_uncertainty.json"))
    args = parser.parse_args()

    current = (DEFAULT_CONFIG.sufficiency.dense_top1, DEFAULT_CONFIG.sufficiency.uncertain_high)
    rows = []
    print(f"{'dense_top1':>10} {'unc_high':>8} {'unc_P':>6} {'unc_R':>6} {'unc_F1':>6} {'supp_P':>6} {'supp_R':>6} {'subR':>6} {'ground':>6}")
    for dense in DENSE_TOP1:
        for high in UNCERTAIN_HIGH:
            overrides = {"sufficiency.dense_top1": dense, "sufficiency.uncertain_high": high}
            runner = ReplayRunner(args.scenarios, mode="ours", split=args.split, config_overrides=overrides)
            m = MetricCalculator.calculate_all(runner._execute_scenarios("ours", runner.load()))
            row = {"dense_top1": dense, "uncertain_high": high, "uncertainty_precision": m["uncertainty_precision"],
                   "uncertainty_recall": m["uncertainty_recall"],
                   "uncertainty_f1": f1(m["uncertainty_precision"], m["uncertainty_recall"]),
                   "suppression_precision": m["suppression_precision"], "suppression_recall": m["suppression_recall"],
                   "sub_intent_recall": m["sub_intent_recall"], "groundedness": m["groundedness"]}
            rows.append(row)
            print(f"{dense:>10.3f} {high:>8.2f} {row['uncertainty_precision']:>6.3f} {row['uncertainty_recall']:>6.3f} "
                  f"{row['uncertainty_f1']:>6.3f} {row['suppression_precision']:>6.3f} {row['suppression_recall']:>6.3f} "
                  f"{row['sub_intent_recall']:>6.3f} {(row['groundedness'] or 0):>6.3f}")

    best = max(rows, key=lambda r: (round(r["uncertainty_f1"], 6), round(r["sub_intent_recall"], 6),
                                    -abs(r["dense_top1"] - current[0]) - abs(r["uncertain_high"] - current[1])))
    print(f"chosen: dense_top1={best['dense_top1']} uncertain_high={best['uncertain_high']} "
          f"F1={best['uncertainty_f1']:.3f} (P={best['uncertainty_precision']:.3f}, R={best['uncertainty_recall']:.3f})")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": args.split, "objective": "uncertainty F1, then sub_intent_recall",
                               "chosen": best, "grid": rows}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
