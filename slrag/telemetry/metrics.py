"""Metrics collector tracking event counts, turn counts, and latency percentiles (p50, p95)."""

from collections import defaultdict
import numpy as np
import threading
import time
from typing import Any, Dict, List

from slrag.contracts.events import MetricsSnapshot
from slrag.replay.metrics import MetricCalculator, calculate_percentile  # noqa: F401  (replay metrics live in slrag.replay)


class MetricsCollector:
    """Thread-safe and async-safe metrics collector."""

    def __init__(self):
        self._lock = threading.Lock()
        self._events_by_type = defaultdict(int)
        self._total_events = 0
        self._total_turns = 0
        self._latencies_ms: List[float] = []
        self._retrieval_latencies_ms: List[float] = []
        self._turn_latencies_ms: List[float] = []
        self._active_sessions = 0
        self._start_time = time.time()

    def record_event(self, event_type: str):
        with self._lock:
            self._events_by_type[event_type] += 1
            self._total_events += 1

    def record_latency(self, latency_ms: float, kind: str = "turn"):
        with self._lock:
            self._latencies_ms.append(latency_ms)
            if kind == "retrieval":
                self._retrieval_latencies_ms.append(latency_ms)
            elif kind == "turn":
                self._turn_latencies_ms.append(latency_ms)

    def record_turn(self):
        with self._lock:
            self._total_turns += 1

    def set_active_sessions(self, count: int):
        with self._lock:
            self._active_sessions = count

    def get_snapshot(self) -> MetricsSnapshot:
        with self._lock:
            if self._latencies_ms:
                p50 = float(np.percentile(self._latencies_ms, 50))
                p95 = float(np.percentile(self._latencies_ms, 95))
            else:
                p50 = 0.0
                p95 = 0.0

            return MetricsSnapshot(
                total_events=self._total_events,
                total_turns=self._total_turns,
                active_sessions=self._active_sessions,
                p50_latency_ms=round(p50, 2),
                p95_latency_ms=round(p95, 2),
                events_by_type=dict(self._events_by_type),
            )

    def get_prometheus_metrics(self) -> str:
        """Format metrics in Prometheus text format."""
        snap = self.get_snapshot()
        lines = [
            "# HELP slrag_events_total Total telemetry events processed",
            "# TYPE slrag_events_total counter",
            f"slrag_events_total {snap.total_events}",
            "# HELP slrag_turns_total Total interaction turns processed",
            "# TYPE slrag_turns_total counter",
            f"slrag_turns_total {snap.total_turns}",
            "# HELP slrag_active_sessions Number of active sessions",
            "# TYPE slrag_active_sessions gauge",
            f"slrag_active_sessions {snap.active_sessions}",
            "# HELP slrag_latency_p50_ms 50th percentile latency in milliseconds",
            "# TYPE slrag_latency_p50_ms gauge",
            f"slrag_latency_p50_ms {snap.p50_latency_ms}",
            "# HELP slrag_latency_p95_ms 95th percentile latency in milliseconds",
            "# TYPE slrag_latency_p95_ms gauge",
            f"slrag_latency_p95_ms {snap.p95_latency_ms}",
        ]
        for evt_type, count in snap.events_by_type.items():
            lines.append(f'slrag_events_by_type{{type="{evt_type}"}} {count}')
        return "\n".join(lines) + "\n"


GLOBAL_METRICS = MetricsCollector()
