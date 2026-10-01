"""Ollama backend: request shape, response parsing and unavailability, with httpx mocked (no Ollama needed)."""

import asyncio

import httpx
import pytest

from slrag.config import DEFAULT_CONFIG, LLMConfig
from slrag.llm import HeuristicLLM, LLMUnavailableError, OllamaClient, get_llm


@pytest.fixture(autouse=True)
def closed_circuit():
    OllamaClient.reset_circuit()
    yield
    OllamaClient.reset_circuit()


def test_chat_posts_to_api_chat_and_parses_text_and_token_counts(monkeypatch):
    calls = []

    async def fake_post(self, url, json=None, **kwargs):
        calls.append({"url": url, "json": json, "timeout": self.timeout})
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "model": "qwen2.5:3b", "message": {"role": "assistant", "content": "The Growth plan costs $199 per month [nx-pricing§growth]."},
            "done": True, "prompt_eval_count": 412, "eval_count": 23,
        })

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    client = get_llm(LLMConfig(backend="ollama", ollama_url="http://localhost:11434", model="qwen2.5:3b"))
    assert isinstance(client, OllamaClient)
    messages = [{"role": "system", "content": "Answer from the context."}, {"role": "user", "content": "What does Growth cost?"}]
    result = asyncio.run(client.chat(messages))

    assert result == {"text": "The Growth plan costs $199 per month [nx-pricing§growth].", "tokens_in": 412, "tokens_out": 23}
    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == "http://localhost:11434/api/chat"
    assert call["json"]["model"] == "qwen2.5:3b" and call["json"]["messages"] == messages and call["json"]["stream"] is False
    assert call["timeout"].read == 10.0  # llm.timeout_s default

    # Structured output for the delta planner / rewriter: the schema goes in Ollama's `format`.
    schema = {"type": "object", "required": ["relation"], "properties": {"relation": {"enum": ["adds"]}}}
    asyncio.run(client.json_fn("plan this", schema, {}))
    assert calls[1]["json"]["format"] == schema and calls[1]["json"]["messages"] == [{"role": "user", "content": "plan this"}]


def test_unreachable_ollama_raises_and_the_default_backend_needs_no_server(monkeypatch):
    async def refused(self, url, json=None, **kwargs):
        raise httpx.ConnectError("connection refused", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", refused)
    with pytest.raises(LLMUnavailableError):
        asyncio.run(OllamaClient("localhost:11434").chat([{"role": "user", "content": "hi"}]))
    # Circuit breaker: the next call fails fast without another connection attempt.
    monkeypatch.setattr(httpx.AsyncClient, "post", lambda *a, **k: pytest.fail("should not connect while the circuit is open"))
    with pytest.raises(LLMUnavailableError, match="marked unavailable"):
        asyncio.run(OllamaClient().chat([{"role": "user", "content": "hi"}]))
    OllamaClient.reset_circuit()

    async def server_error(self, url, json=None, **kwargs):
        return httpx.Response(404, request=httpx.Request("POST", url), json={"error": "model 'qwen2.5:3b' not found"})

    monkeypatch.setattr(httpx.AsyncClient, "post", server_error)
    with pytest.raises(LLMUnavailableError, match="HTTP 404"):
        asyncio.run(OllamaClient().chat([{"role": "user", "content": "hi"}]))

    assert DEFAULT_CONFIG.llm.backend == "heuristic" and isinstance(get_llm(), HeuristicLLM)


def test_drafter_uses_ollama_and_falls_back_when_it_is_unreachable(monkeypatch):
    from slrag.pipeline.llm import LLMServiceWrapper

    chunks = [{"chunk_id": "nx-pricing§growth", "text": "The Growth plan costs $199 per month and is the most popular plan."}]

    def answer(self, url, json=None, **kwargs):
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "message": {"content": "The Growth plan costs $199 per month [nx-pricing§growth]. It is popular [nx-pricing§growth]."},
            "prompt_eval_count": 90, "eval_count": 20,
        })

    monkeypatch.setattr(httpx.Client, "post", answer)
    drafter = LLMServiceWrapper(LLMConfig(backend="ollama"))
    result = drafter.generate_grounded_response_sync("What does Growth cost?", chunks)
    assert drafter.last_mode == "ollama" and (result.tokens_in, result.tokens_out) == (90, 20)
    assert result.raw_claims == [("The Growth plan costs $199 per month.", ["nx-pricing§growth"]), ("It is popular.", ["nx-pricing§growth"])]

    def refused(self, url, json=None, **kwargs):
        raise httpx.ConnectError("connection refused", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", refused)
    fallback = drafter.generate_grounded_response_sync("What does Growth cost?", chunks)
    assert drafter.last_mode == "heuristic_fallback" and fallback.raw_claims  # extractive draft, the turn still answers
