"""Ollama backend for the LLM roles (llm.backend: ollama).

Talks to a local Ollama server's chat API (POST {ollama_url}/api/chat, stream: false) and returns
{text, tokens_in, tokens_out} from the response (message.content, prompt_eval_count, eval_count).
Every call has a timeout (llm.timeout_s, default 10 s); connection failures, timeouts and HTTP
errors raise LLMUnavailableError so callers can fall back instead of hanging a live turn.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx

Message = Dict[str, str]


class LLMUnavailableError(RuntimeError):
    """The configured LLM backend could not be reached or did not answer usably."""


def _base_url(ollama_url: str) -> str:
    url = ollama_url.strip().rstrip("/")
    return url if url.startswith(("http://", "https://")) else f"http://{url}"


class OllamaClient:
    mode = "ollama"
    # Circuit breaker, shared per URL: after a connection failure, calls fail fast for this long instead of
    # paying a connect attempt each time (a refused localhost connect costs ~2 s on Windows), so a live
    # turn falls back promptly while Ollama is down.
    retry_after_s = 30.0
    _down_until: Dict[str, float] = {}

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "qwen2.5:3b",
                 timeout_s: float = 10.0, temperature: float = 0.0):
        self.url = f"{_base_url(ollama_url)}/api/chat"
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature

    @classmethod
    def reset_circuit(cls) -> None:
        cls._down_until.clear()

    def _check_circuit(self) -> None:
        until = self._down_until.get(self.url, 0.0)
        if time.monotonic() < until:
            raise LLMUnavailableError(f"Ollama at {self.url} marked unavailable for another {until - time.monotonic():.0f}s")

    def _trip(self, exc: Exception) -> LLMUnavailableError:
        self._down_until[self.url] = time.monotonic() + self.retry_after_s
        return LLMUnavailableError(f"Ollama not reachable at {self.url}: {exc!r}")

    def _payload(self, messages: List[Message], fmt: Optional[Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if fmt is not None:
            payload["format"] = fmt  # "json" or a JSON schema (Ollama structured outputs)
        return payload

    @staticmethod
    def _parse(response: httpx.Response) -> Dict[str, Any]:
        try:
            response.raise_for_status()
            data = response.json()
            text = data["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise LLMUnavailableError(f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text[:200]}") from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMUnavailableError(f"unexpected Ollama response: {exc!r}") from exc
        return {"text": text, "tokens_in": int(data.get("prompt_eval_count") or 0), "tokens_out": int(data.get("eval_count") or 0)}

    async def chat(self, messages: List[Message], fmt: Optional[Any] = None) -> Dict[str, Any]:
        self._check_circuit()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                response = await client.post(self.url, json=self._payload(messages, fmt))
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
            raise self._trip(exc) from exc
        return self._parse(response)

    def chat_sync(self, messages: List[Message], fmt: Optional[Any] = None) -> Dict[str, Any]:
        """Blocking variant for callers that already run in a worker thread (the drafter)."""
        self._check_circuit()
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                response = client.post(self.url, json=self._payload(messages, fmt))
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
            raise self._trip(exc) from exc
        return self._parse(response)

    # -- JsonFn for json_call (Phase 7 delta planner / rewriter) --------------------------

    async def json_fn(self, prompt: str, schema: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Schema-constrained JSON: the enum-bearing schema is passed as Ollama's `format`."""
        result = await self.chat([{"role": "user", "content": prompt}], fmt=schema)
        return result["text"]

    @property
    def delta_fn(self):
        return self.json_fn

    @property
    def rewrite_fn(self):
        return self.json_fn
