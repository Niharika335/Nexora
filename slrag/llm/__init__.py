"""LLM backends and structured-output calls: get_llm(), enum-constrained schemas, prompts, json_call."""

from typing import Any, Optional, Union

from slrag.llm.heuristic import HeuristicLLM
from slrag.llm.json_call import JsonCallError, JsonCallResult, JsonFn, estimate_tokens, json_call
from slrag.llm.ollama_client import LLMUnavailableError, OllamaClient
from slrag.llm.schemas import SchemaError, delta_planner_schema, rewriter_schema, validate


def get_llm(config: Optional[Any] = None) -> Union[OllamaClient, HeuristicLLM]:
    """OllamaClient when llm.backend == "ollama", HeuristicLLM otherwise (the default)."""
    if config is None:
        from slrag.config import DEFAULT_CONFIG

        config = DEFAULT_CONFIG.llm
    if getattr(config, "backend", "heuristic") == "ollama":
        return OllamaClient(config.ollama_url, config.model, timeout_s=config.timeout_s, temperature=config.temperature)
    return HeuristicLLM()

__all__ = [
    "HeuristicLLM",
    "LLMUnavailableError",
    "OllamaClient",
    "get_llm",
    "JsonCallError",
    "JsonCallResult",
    "JsonFn",
    "SchemaError",
    "delta_planner_schema",
    "estimate_tokens",
    "json_call",
    "rewriter_schema",
    "validate",
]
