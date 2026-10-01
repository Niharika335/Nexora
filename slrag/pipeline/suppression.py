"""Per-sub-intent sufficiency decision: answer, answer with uncertainty, or suppress (no claim)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from slrag.contracts.events import SubIntentCompletionEvent


@dataclass
class GateOutcome:
    sufficient: bool
    score: float
    reason: str = ""
    dense_top1: Optional[float] = None
    coverage: Optional[float] = None


def normalize_gate_result(result: Any) -> GateOutcome:
    """Accept both SufficiencyGate.evaluate tuples (passed, dense_top1, coverage, reason) and result objects."""
    if isinstance(result, tuple):
        passed, dense_top1, coverage, reason = result
        return GateOutcome(bool(passed), float(coverage), str(reason), float(dense_top1), float(coverage))
    dense = getattr(result, "dense_top1", None)
    cov = getattr(result, "coverage", None)
    return GateOutcome(
        bool(getattr(result, "sufficient", False)),
        float(getattr(result, "score", 0.0)),
        str(getattr(result, "reason", "")),
        None if dense is None else float(dense),
        None if cov is None else float(cov),
    )


def gate_thresholds(gate: Any) -> Dict[str, float]:
    """{dense_top1, coverage} thresholds of a SufficiencyGate or an adapter wrapping one (telemetry only)."""
    cfg = getattr(getattr(gate, "gate", gate), "config", None)
    if cfg is None or not hasattr(cfg, "dense_top1"):
        return {}
    return {"dense_top1": float(cfg.dense_top1), "coverage": float(cfg.coverage)}


def gate_fields(outcome: GateOutcome, gate: Any) -> Dict[str, Any]:
    """Telemetry fields describing a gate outcome (scores + thresholds)."""
    return {
        "dense_top1": None if outcome.dense_top1 is None else round(outcome.dense_top1, 4),
        "coverage": None if outcome.coverage is None else round(outcome.coverage, 4),
        "thresholds": gate_thresholds(gate),
    }


class SuppressionController:
    def __init__(
        self,
        sufficiency_gate: Any,
        bus: Optional[Any] = None,
        clock: Optional[Any] = None,
        uncertain_band: tuple = (0.50, 0.70),
    ):
        self.sufficiency_gate = sufficiency_gate
        self.bus = bus
        self.clock = clock
        self.uncertain_band = uncertain_band

    def _publish(self, **kw: Any) -> None:
        if self.bus:
            if self.clock:
                kw["timestamp"] = self.clock.time()
            self.bus.publish(SubIntentCompletionEvent(**kw))

    def evaluate(
        self,
        sub_intent_id: str,
        sub_query: str,
        chunks: List[Any],
        turn_id: str = "",
        is_unanswerable_ground_truth: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Suppressed sub-intents get an uncertainty item and no claim; answerable ones in the
        marginal score band are answered but flagged uncertain."""
        base = {"turn_id": turn_id, "sub_intent_id": sub_intent_id, "is_unanswerable_ground_truth": is_unanswerable_ground_truth}

        if not chunks:
            self._publish(**base, status="suppressed", is_suppressed=True, is_uncertain=True, reason="no_evidence")
            return {"suppressed": True, "uncertain": True, "score": 0.0, "reason": "no_evidence",
                    "text": "Insufficient evidence: no relevant passages retrieved."}

        outcome = normalize_gate_result(self.sufficiency_gate.evaluate(sub_query, chunks))
        base.update(gate_fields(outcome, self.sufficiency_gate))
        if not outcome.sufficient:
            self._publish(**base, status="suppressed", is_suppressed=True, is_uncertain=True, reason="insufficient_evidence")
            return {"suppressed": True, "uncertain": True, "score": outcome.score, "reason": "insufficient_evidence",
                    "text": f"Suppressed due to insufficient context: {outcome.reason}"}

        low, high = self.uncertain_band
        is_uncertain = low <= outcome.score < high
        self._publish(**base, status="uncertain" if is_uncertain else "completed", is_suppressed=False, is_uncertain=is_uncertain)
        return {"suppressed": False, "uncertain": is_uncertain, "score": outcome.score, "reason": "", "text": ""}
