"""T2: LLM RETRIEVE / NO_RETRIEVE classifier for the ambiguous T0 band."""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional, Tuple

CHOICES = ("RETRIEVE", "NO_RETRIEVE")
ClassifyFn = Callable[[str], Awaitable[str]]


class T2Classifier:
    """Wraps an async classify call with a hard timeout. Any failure falls back to RETRIEVE."""

    def __init__(self, classify_fn: Optional[ClassifyFn] = None, timeout_s: float = 0.4):
        self.classify_fn = classify_fn
        self.timeout_s = timeout_s

    async def decide(self, buffer: str) -> Tuple[str, str]:
        """Return (choice, reason)."""
        if self.classify_fn is None:
            return "RETRIEVE", "t2_unavailable"
        try:
            choice = await asyncio.wait_for(self.classify_fn(buffer), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            return "RETRIEVE", "t2_timeout"
        except Exception:
            return "RETRIEVE", "t2_error"
        choice = str(choice).strip().upper()
        if choice not in CHOICES:
            return "RETRIEVE", "t2_invalid"
        return choice, "t2_classified"
