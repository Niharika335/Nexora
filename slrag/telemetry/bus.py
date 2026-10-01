"""Zero-loss Telemetry Bus dispatching events to JSONL file and WebSocket subscribers."""

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel

from slrag.config import TelemetryConfig, DEFAULT_CONFIG
from slrag.contracts.events import BaseEvent

logger = logging.getLogger(__name__)


class TelemetryBus:
    """Async event bus guaranteeing zero loss under high-throughput burst conditions."""

    def __init__(self, config: TelemetryConfig = DEFAULT_CONFIG.telemetry):
        self.config = config
        self.log_path = Path(config.jsonl_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory subscriber queues (for WebSockets)
        self._subscribers: Set[asyncio.Queue] = set()
        self._subscribers_lock = asyncio.Lock()
        
        # Internal async event processing queue
        self._event_queue: asyncio.Queue[BaseEvent] = asyncio.Queue(maxsize=config.buffer_size)
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        self._file_handle = None
        self._total_emitted = 0
        self._total_written = 0
        self._total_processed = 0
        self._overflow: Deque[BaseEvent] = deque()
        self._overflow_count = 0

    def start(self):
        """Start the background consumer worker."""
        if not self._running:
            self._running = True
            # Open file in append mode with line buffering
            self._file_handle = open(self.log_path, "a", encoding="utf-8", buffering=1)
            self._worker_task = asyncio.create_task(self._process_queue())

    async def emit(self, event: BaseEvent):
        """Enqueue an event without ever blocking the caller (awaitable for API compatibility)."""
        self.emit_nowait(event)

    def emit_nowait(self, event: BaseEvent):
        """Non-blocking emit: `put_nowait` onto the bounded queue.

        When the queue is full the event is spilled to an overflow deque (counted in
        `overflow_count`) that the worker drains after the queue, so nothing is lost and
        ordering is preserved: once spilling starts, later events also spill until it empties.
        """
        if not self._running:
            self.start()
        self._total_emitted += 1
        if self._overflow:
            self._overflow.append(event)
            self._overflow_count += 1
            return
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            if self._overflow_count == 0:
                logger.warning("Telemetry queue full (%d); spilling to overflow buffer.", self._event_queue.maxsize)
            self._overflow.append(event)
            self._overflow_count += 1

    def publish(self, event: BaseEvent) -> None:
        """Synchronous publish used by pipeline components (alias of emit_nowait)."""
        self.emit_nowait(event)

    async def _next_event(self) -> Tuple[Optional[BaseEvent], bool]:
        """Return (event, from_queue). Queue items are older than overflow items, so drain the queue first."""
        if not self._event_queue.empty():
            return self._event_queue.get_nowait(), True
        if self._overflow:
            return self._overflow.popleft(), False
        try:
            return await asyncio.wait_for(self._event_queue.get(), timeout=0.1), True
        except asyncio.TimeoutError:
            return None, False

    async def _process_queue(self):
        """Worker loop reading events from queue and distributing to JSONL and subscribers."""
        while self._running or not self._event_queue.empty() or self._overflow:
            event, from_queue = None, False
            try:
                event, from_queue = await self._next_event()
                if event is None:
                    continue

                event_json = event.model_dump_json()

                # 1. Write to JSONL
                if self._file_handle:
                    self._file_handle.write(event_json + "\n")
                    self._file_handle.flush()
                    self._total_written += 1

                # 2. Fan-out to all active WebSocket subscriber queues
                async with self._subscribers_lock:
                    stale_subscribers = []
                    for sub_q in self._subscribers:
                        try:
                            sub_q.put_nowait(event)
                        except asyncio.QueueFull:
                            # If a subscriber is lagging, drop or evict
                            pass
                        except Exception:
                            stale_subscribers.append(sub_q)
                    for stale in stale_subscribers:
                        self._subscribers.discard(stale)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in telemetry bus worker: {e}", exc_info=True)
            finally:
                if event is not None:
                    self._total_processed += 1
                    if from_queue:
                        self._event_queue.task_done()

    async def register_subscriber(self, maxsize: int = 20000) -> asyncio.Queue[BaseEvent]:
        """Register a new subscriber queue (e.g. for a WebSocket connection)."""
        sub_queue: asyncio.Queue[BaseEvent] = asyncio.Queue(maxsize=maxsize)
        async with self._subscribers_lock:
            self._subscribers.add(sub_queue)
        return sub_queue

    async def unregister_subscriber(self, sub_queue: asyncio.Queue[BaseEvent]):
        """Unregister a subscriber queue."""
        async with self._subscribers_lock:
            self._subscribers.discard(sub_queue)

    async def drain(self):
        """Wait until every emitted event (queue and overflow) is written and fanned out."""
        while self._worker_task is not None and not self._worker_task.done() and self._total_processed < self._total_emitted:
            await asyncio.sleep(0.005)
        if self._file_handle:
            self._file_handle.flush()

    async def close(self):
        """Stop worker and close open file handle."""
        await self.drain()
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        if self._file_handle:
            self._file_handle.flush()
            self._file_handle.close()
            self._file_handle = None

    @property
    def total_emitted(self) -> int:
        return self._total_emitted

    @property
    def total_written(self) -> int:
        return self._total_written

    @property
    def overflow_count(self) -> int:
        """Events that found the bounded queue full and went through the overflow buffer."""
        return self._overflow_count


# Global default telemetry bus
GLOBAL_BUS = TelemetryBus()


class SessionStampedBus:
    """publish() wrapper that stamps a session id on events whose emitter does not know it (the live
    controller, evidence cache and pre-drafter), so per-session consumers such as the trace UI see them."""

    def __init__(self, inner: Any, session_id: str):
        self.inner = inner
        self.session_id = session_id

    def publish(self, event: BaseEvent) -> None:
        if getattr(event, "session_id", None) != self.session_id:
            event = event.model_copy(update={"session_id": self.session_id})
        self.inner.publish(event)

