"""json_call: one bounded, schema-validated structured-output LLM call."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import re
from typing import Any, Awaitable, Callable, Dict, Optional

from slrag.llm.schemas import SchemaError, validate

# An LLM backend: (prompt, schema, context) -> JSON text or an already-parsed object.
# `context` carries the same inputs as the prompt in structured form; a real model can ignore it.
JsonFn = Callable[[str, Dict[str, Any], Dict[str, Any]], Awaitable[Any]]

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens(text: str) -> int:
    """Same BPE-style estimate as LLMServiceWrapper.estimate_tokens."""
    return max(1, len(_TOKEN_RE.findall(text)))


class JsonCallError(Exception):
    """kind: timeout | invalid_json | schema | llm_error."""

    def __init__(self, kind: str, message: str, tokens_in: int = 0):
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.tokens_in = tokens_in


@dataclass
class JsonCallResult:
    data: Dict[str, Any]
    raw: str
    tokens_in: int
    tokens_out: int
    latency_ms: float


async def json_call(
    fn: JsonFn,
    prompt: str,
    schema: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    timeout_s: float = 1.5,
) -> JsonCallResult:
    """Call `fn` once with a timeout; parse its JSON and validate it against `schema`.

    Raises JsonCallError on timeout, unparseable JSON, a schema violation or any backend error.
    There are no retries: the caller applies its own fallback.
    """
    tokens_in = estimate_tokens(prompt)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    try:
        raw = await asyncio.wait_for(fn(prompt, schema, context or {}), timeout=timeout_s)
    except asyncio.TimeoutError:
        raise JsonCallError("timeout", f"no response within {timeout_s}s", tokens_in) from None
    except Exception as exc:  # backend failure
        raise JsonCallError("llm_error", repr(exc), tokens_in) from exc

    raw_text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as exc:
        raise JsonCallError("invalid_json", str(exc), tokens_in) from None
    try:
        validate(data, schema)
    except SchemaError as exc:
        raise JsonCallError("schema", str(exc), tokens_in) from None
    return JsonCallResult(data, raw_text, tokens_in, estimate_tokens(raw_text), (loop.time() - t0) * 1000.0)
