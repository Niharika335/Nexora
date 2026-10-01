"""Token-level cascade trigger: confidence/entropy heuristic over streamed tokens.

This is the lightweight per-token speculation trigger. The chunk-level T0/T1/T2 retrieval
controller used by the streaming turn engine lives in `slrag.control.controller`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from slrag.contracts.events import CascadeTriggerEvent, SpeculationOutcomeEvent
from slrag.nlp.lemmas import content_tokens

BOUNDARY_CUES = (".", "?", ",", "and", "or", "what", "how", "who", "when", "explain", "summarize")


class CascadeState:
    IDLE = "IDLE"
    ACCUMULATING = "ACCUMULATING"
    SPECULATING = "SPECULATING"
    STABLE = "STABLE"
    INVALIDATED = "INVALIDATED"


class CascadeController:
    def __init__(
        self,
        config: Optional[Any] = None,
        bus: Optional[Any] = None,
        clock: Optional[Any] = None,
        confidence_threshold: Optional[float] = None,
        entropy_threshold: Optional[float] = None,
        min_token_boundary: Optional[int] = None,
    ):
        """Thresholds come from explicit kwargs, else `config` (SlragConfig or CascadeConfig field names), else defaults."""
        def pick(explicit: Optional[Any], names: tuple, default: Any) -> Any:
            if explicit is not None:
                return explicit
            for name in names:
                if config is not None and hasattr(config, name):
                    return getattr(config, name)
            return default

        self.confidence_threshold = pick(confidence_threshold, ("cascade_confidence_threshold", "confidence_threshold"), 0.72)
        self.entropy_threshold = pick(entropy_threshold, ("cascade_entropy_threshold", "entropy_threshold"), 0.38)
        self.min_token_boundary = pick(min_token_boundary, ("min_token_boundary",), 4)
        self.bus = bus
        self.clock = clock
        self.state = CascadeState.IDLE
        self.tokens: List[str] = []
        self.speculated_query: Optional[str] = None
        self.speculation_used = False

    def reset(self, turn_id: str = "") -> None:
        self.state = CascadeState.IDLE
        self.tokens = []
        self.speculated_query = None
        self.speculation_used = False
        self._emit(turn_id, 0, 0.0, 0.0, "reset")

    def _publish(self, event: Any) -> None:
        if self.bus:
            self.bus.publish(event)

    def _ts(self) -> Dict[str, Any]:
        return {"timestamp": self.clock.time()} if self.clock else {}

    def _emit(self, turn_id: str, token_idx: int, conf: float, ent: float, trigger_type: str) -> None:
        self._publish(CascadeTriggerEvent(
            turn_id=turn_id, token_index=token_idx, confidence=conf, entropy=ent,
            state=self.state, trigger_type=trigger_type, **self._ts(),
        ))

    def evaluate_token(self, token: str, turn_id: str = "") -> Dict[str, Any]:
        self.tokens.append(token)
        token_count = len(self.tokens)
        current_text = "".join(self.tokens).strip()

        if token_count < self.min_token_boundary:
            self.state = CascadeState.ACCUMULATING
            self._emit(turn_id, token_count, 0.2, 0.8, "accumulating")
            return {"trigger": False}

        has_cue = any(token.strip().lower().endswith(c) for c in BOUNDARY_CUES)
        confidence = min(1.0, 0.5 + 0.05 * token_count + (0.2 if has_cue else 0.0))
        entropy = max(0.0, 1.0 - confidence)

        if confidence >= self.confidence_threshold and entropy <= self.entropy_threshold and self.state != CascadeState.SPECULATING:
            self.state = CascadeState.SPECULATING
            self.speculated_query = current_text
            self._emit(turn_id, token_count, confidence, entropy, "early_retrieval")
            return {"trigger": True, "query": current_text}

        self._emit(turn_id, token_count, confidence, entropy, "eval")
        return {"trigger": False}

    def finalize_turn(self, final_query: str, turn_id: str = "") -> None:
        """Speculation is `used` if its content tokens are a prefix of the final query's (or vice versa)."""
        if self.state != CascadeState.SPECULATING:
            return
        if not self.speculated_query:
            self.state = CascadeState.INVALIDATED
            self._publish(SpeculationOutcomeEvent(turn_id=turn_id, outcome="false_trigger", reason="unused_speculation", **self._ts()))
            return

        spec, final = content_tokens(self.speculated_query), content_tokens(final_query)
        shorter, longer = (spec, final) if len(spec) <= len(final) else (final, spec)
        if shorter and longer[: len(shorter)] == shorter:
            self.state = CascadeState.STABLE
            self.speculation_used = True
            self._publish(SpeculationOutcomeEvent(turn_id=turn_id, outcome="used", reason="speculative_match", **self._ts()))
        else:
            self.state = CascadeState.INVALIDATED
            self._publish(SpeculationOutcomeEvent(turn_id=turn_id, outcome="invalidated", reason="query_diverged", **self._ts()))
