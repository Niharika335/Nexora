"""Deterministic replay virtual clock and latency model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


class ReplayClock:
    def __init__(self, start_time: float = 1000.0, tick_interval: float = 0.050):
        self._current_time = start_time
        self._tick_interval = tick_interval

    def time(self) -> float:
        return self._current_time

    def advance(self, delta: Optional[float] = None) -> float:
        d = self._tick_interval if delta is None else delta
        self._current_time += d
        return self._current_time

    def advance_to(self, t: float) -> float:
        if t > self._current_time:
            self._current_time = t
        return self._current_time

    def reset(self, start_time: float = 1000.0) -> None:
        self._current_time = start_time


@dataclass(frozen=True)
class LatencyModel:
    """Virtual-time cost of each pipeline stage (seconds), used only under a ReplayClock.

    Retrieval dispatched before utterance_end runs concurrently with speech, so only its
    unfinished remainder is charged when the final path waits on it. Sub-query retrievals of
    one wave run in parallel (charged once). Per-sub-intent drafts run in parallel, so the
    first emitted sub-intent pays draft + verify and later ones only verify.
    """
    retrieval_s: float = 0.120
    cache_hit_s: float = 0.002
    plan_s: float = 0.005
    draft_s: float = 0.350
    verify_s: float = 0.015
    present_s: float = 0.020
    chunk_interval_s: float = 0.200
    utterance_end_gap_s: float = 0.100
    words_per_chunk: int = 3
    reconcile_s: float = 0.003  # Phase 8: evidence comparison at utterance_end (CPU only)


DEFAULT_LATENCY = LatencyModel()


def charge(clock: Optional[Any], seconds: float) -> None:
    """Advance a virtual clock; a no-op for wall-clock runs (clock is None or has no advance)."""
    if clock is not None and hasattr(clock, "advance"):
        clock.advance(seconds)
