"""Threshold calibration sweep over the tune split only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from slrag.config import DEFAULT_CONFIG, SlragConfig
from slrag.eval.runner import ReplayRunner
from slrag.eval.scenarios import write_split_files

# Controller drift threshold x sufficiency coverage threshold (dotted keys override AppConfig sections).
CANDIDATES = [
    {"controller.new_info_cos": 0.78, "sufficiency.coverage": 0.45},
    {"controller.new_info_cos": 0.82, "sufficiency.coverage": 0.50},
    {"controller.new_info_cos": 0.86, "sufficiency.coverage": 0.50},
    {"controller.new_info_cos": 0.82, "sufficiency.coverage": 0.60},
]


class CalibrationSweep:
    @staticmethod
    def run_sweep(tune_scenarios_path: str = "eval/scenarios/tune.jsonl", out_dir: str = "out") -> Dict[str, Any]:
        p = Path(tune_scenarios_path)
        if "test" in str(p).lower():
            raise ValueError("Test scenarios must NEVER be used for calibration.")

        if not p.exists():
            write_split_files(str(p.parent))

        # Objective: sub-intent recall + suppression recall - wasted speculation.
        best_score, best_cfg, results = -1.0, CANDIDATES[1], []
        for cand in CANDIDATES:
            runner = ReplayRunner(
                scenarios_path=str(p), mode="ours", out_path=str(Path(out_dir) / "tune_telemetry.jsonl"),
                playback="virtual", config_overrides=cand,
            )
            metrics = runner.run(generate_report=False)["metrics"]
            score = (
                (metrics.get("subintent_recall") or 0.0)
                + (metrics.get("suppression_recall") or 0.0)
                - (metrics.get("wasted_speculation_rate") or 0.0)
            )
            results.append({"candidate": cand, "score": round(score, 4)})
            if score > best_score:
                best_score, best_cfg = score, cand

        defaults = SlragConfig()
        final_calibrated = {
            "seed": 13,
            "temperature": 0.0,
            "cascade_confidence_threshold": defaults.cascade_confidence_threshold,
            "cascade_entropy_threshold": defaults.cascade_entropy_threshold,
            "min_token_boundary": defaults.min_token_boundary,
            "sufficiency_threshold": best_cfg["sufficiency.coverage"],
            "suppression_threshold": defaults.suppression_threshold,
            "controller.new_info_cos": best_cfg["controller.new_info_cos"],
            "sufficiency.coverage": best_cfg["sufficiency.coverage"],
            "sufficiency.dense_top1": DEFAULT_CONFIG.sufficiency.dense_top1,
            "enable_verifier": True,
            "enable_cascade": True,
            "enable_multi_intent": True,
            "sweep": results,
        }

        cal_out = Path(out_dir) / "calibrated_config.json"
        cal_out.parent.mkdir(parents=True, exist_ok=True)
        with open(cal_out, "w", encoding="utf-8") as f:
            json.dump(final_calibrated, f, indent=2)

        return final_calibrated
