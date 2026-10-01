"""Sufficiency Gate checking retrieval score thresholds (dense_top1 >= 0.55, coverage >= 0.50)."""

from dataclasses import dataclass
import re
from typing import List, Optional, Tuple
from slrag.config import SufficiencyConfig, DEFAULT_CONFIG
from slrag.contracts.events import SufficiencyCheckEvent
from slrag.retrieval.engine import SearchResult
from slrag.telemetry.bus import TelemetryBus, GLOBAL_BUS


def compute_query_coverage(query: str, retrieved_texts: List[str]) -> float:
    """Calculate the ratio of informative query keywords present in retrieved text."""
    stop_words = {
        "a", "an", "the", "in", "on", "of", "to", "for", "is", "are", "was", "were",
        "and", "or", "what", "how", "why", "who", "which", "where", "can", "does",
        "do", "explain", "tell", "me", "about", "describe", "with", "from", "at"
    }
    q_tokens = [w.lower() for w in re.findall(r"\b\w+\b", query) if w.lower() not in stop_words]
    if not q_tokens:
        return 1.0

    combined_text = " ".join(retrieved_texts).lower()
    matched = sum(1 for tok in q_tokens if tok in combined_text)
    return matched / len(q_tokens)


class SufficiencyGate:
    """Gatekeeper verifying whether retrieved context is sufficient for reliable answer generation."""

    def __init__(self, config: SufficiencyConfig = DEFAULT_CONFIG.sufficiency):
        self.config = config

    def evaluate(self, query: str, results: List[SearchResult]) -> Tuple[bool, float, float, str]:
        """Evaluate retrieval sufficiency against dense_top1 and coverage thresholds.
        
        Returns:
            (passed, dense_top1_score, coverage_score, reason)
        """
        if not results:
            return False, 0.0, 0.0, "No retrieval results returned."

        # Extract dense_top1 score
        dense_top1_score = 0.0
        for r in results:
            if "dense_score" in r.source_scores:
                dense_top1_score = max(dense_top1_score, float(r.source_scores["dense_score"]))
            elif "dense" in r.source_scores:
                dense_top1_score = max(dense_top1_score, float(r.source_scores["dense"]))
            else:
                dense_top1_score = max(dense_top1_score, float(r.score))
            break

        # Extract coverage score
        retrieved_texts = [r.text for r in results[:3]]
        coverage_score = compute_query_coverage(query, retrieved_texts)

        dense_pass = dense_top1_score >= self.config.dense_top1
        coverage_pass = coverage_score >= self.config.coverage

        if dense_pass and coverage_pass:
            return (
                True,
                dense_top1_score,
                coverage_score,
                f"Sufficient: dense_top1={dense_top1_score:.3f} >= {self.config.dense_top1}, coverage={coverage_score:.3f} >= {self.config.coverage}",
            )
        else:
            reasons = []
            if not dense_pass:
                reasons.append(f"dense_top1={dense_top1_score:.3f} < {self.config.dense_top1}")
            if not coverage_pass:
                reasons.append(f"coverage={coverage_score:.3f} < {self.config.coverage}")
            return False, dense_top1_score, coverage_score, f"Insufficient context: {', '.join(reasons)}"

    async def evaluate_and_emit(
        self,
        query: str,
        results: List[SearchResult],
        turn_id: str,
        session_id: str = "default_session",
        seq: int = 0,
        bus: Optional[TelemetryBus] = None,
    ) -> Tuple[bool, float, float, str]:
        """Evaluate gate and emit SufficiencyCheckEvent to telemetry bus."""
        telemetry_bus = bus or GLOBAL_BUS
        passed, dense_score, coverage, reason = self.evaluate(query, results)

        event = SufficiencyCheckEvent(
            session_id=session_id,
            turn_id=turn_id,
            seq=seq,
            dense_top1_score=round(dense_score, 4),
            coverage_score=round(coverage, 4),
            passed=passed,
            reason=reason,
            thresholds={"dense_top1": self.config.dense_top1, "coverage": self.config.coverage},
        )
        await telemetry_bus.emit(event)
        return passed, dense_score, coverage, reason
