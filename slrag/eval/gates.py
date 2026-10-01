"""Evaluation of the G1-G6 acceptance gates."""
from __future__ import annotations

from typing import Any, Dict, Optional

G2_MIN_EARLY_RETRIEVAL_RATE = 0.80
G4_MIN_GROUNDEDNESS = 0.85
G4_MAX_HALLUCINATED_ID_RATE = 0.0
G6_REQUIRED_TRACE_COVERAGE = 1.0
G5_REQUIRED_PRESERVATION_RATE = 1.0


def _status(ok: Optional[bool]) -> str:
    """A gate whose metric is missing fails rather than passing by default."""
    return "PASS" if ok else "FAIL"


class GateEvaluator:
    @staticmethod
    def evaluate_gates(
        ours_metrics: Dict[str, Any],
        b1_metrics: Dict[str, Any],
        b0_metrics: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        ours_r10 = ours_metrics.get("recall@10") or 0.0
        b1_r10 = b1_metrics.get("recall@10") or 0.0

        ours_ground = ours_metrics.get("groundedness")
        b1_ground = b1_metrics.get("groundedness") or 0.0

        early = ours_metrics.get("early_retrieval_rate")
        halluc = ours_metrics.get("hallucinated_id_rate")
        coverage = ours_metrics.get("trace_coverage")
        g5 = GateEvaluator.g5_late_detail(ours_metrics)

        return {
            "G1_recall_improvement": {
                "metric": "recall@10",
                "condition": "Ours >= B1",
                "ours": ours_r10,
                "b1": b1_r10,
                "status": _status(ours_r10 >= b1_r10),
            },
            "G2_early_retrieval": {
                "metric": "early_retrieval_rate",
                "condition": f"Ours >= {G2_MIN_EARLY_RETRIEVAL_RATE}",
                "ours": early,
                "status": _status(early is not None and early >= G2_MIN_EARLY_RETRIEVAL_RATE),
            },
            "G3_groundedness_preservation": {
                "metric": "groundedness",
                "condition": "Ours >= B1",
                "ours": ours_ground or 0.0,
                "b1": b1_ground,
                "status": _status((ours_ground or 0.0) >= b1_ground),
            },
            "G4_grounding": {
                "metric": "groundedness, hallucinated_id_rate",
                "condition": f"groundedness >= {G4_MIN_GROUNDEDNESS} and hallucinated_id_rate == {G4_MAX_HALLUCINATED_ID_RATE}",
                "ours": {"groundedness": ours_ground, "hallucinated_id_rate": halluc},
                "status": _status(
                    ours_ground is not None and ours_ground >= G4_MIN_GROUNDEDNESS
                    and halluc is not None and halluc <= G4_MAX_HALLUCINATED_ID_RATE
                ),
            },
            "G5_late_detail_refinement": g5,
            "G6_trace_coverage": {
                "metric": "trace_coverage",
                "condition": f"Ours == {G6_REQUIRED_TRACE_COVERAGE}",
                "ours": coverage,
                "status": _status(coverage is not None and coverage >= G6_REQUIRED_TRACE_COVERAGE),
            },
        }

    @staticmethod
    def g5_late_detail(m: Dict[str, Any]) -> Dict[str, Any]:
        """G5 passes if preservation_rate = 1.0, no late-detail turn restarted or fully re-searched,
        and version lineage is complete. NOT_EVALUATED when the replay contained no late-detail turn;
        FAIL when the metrics are missing altogether."""
        turns = m.get("refinement_turns")
        preservation = m.get("preservation_rate")
        restarts = m.get("late_restart_turns")
        lineage = m.get("version_lineage_complete")
        gate = {
            "metric": "preservation_rate, late_restart_turns, version_lineage_complete",
            "condition": f"preservation_rate == {G5_REQUIRED_PRESERVATION_RATE}, late_restart_turns == 0, version_lineage_complete == 1.0",
            "ours": {"refinement_turns": turns, "preservation_rate": preservation, "late_restart_turns": restarts,
                     "version_lineage_complete": lineage},
        }
        if turns == 0:
            gate["status"] = "NOT_EVALUATED"
            gate["condition"] += " (no late-detail turns in this replay)"
            return gate
        gate["status"] = _status(
            turns is not None and preservation is not None and preservation >= G5_REQUIRED_PRESERVATION_RATE
            and restarts == 0 and lineage is not None and lineage >= 1.0
        )
        return gate
