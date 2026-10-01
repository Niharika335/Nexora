"""LLM service wrapper for Qwen2.5-3B-Instruct with token and cost tracking."""

from dataclasses import dataclass
import logging
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from slrag.config import LLMConfig, DEFAULT_CONFIG


@dataclass
class LLMGenerationResult:
    text: str
    tokens_in: int
    tokens_out: int
    cost: float
    model_name: str
    raw_claims: List[Tuple[str, List[str]]]  # List of (claim_text, cited_doc_ids)


logger = logging.getLogger(__name__)
CITE_GROUP_RE = re.compile(r"\[([^\[\]]+)\]")


class LLMServiceWrapper:
    """Grounded drafting with token and cost accounting.

    llm.backend heuristic (default): extractive drafting from the retrieved chunks (no model).
    llm.backend ollama: the model drafts from the chunks via Ollama; if Ollama is unreachable the
    extractive path is used for that call and `last_mode` records "heuristic_fallback".
    """

    def __init__(self, config: LLMConfig = DEFAULT_CONFIG.llm):
        self.config = config
        self.last_mode = getattr(config, "backend", "heuristic")
        self._client = None
        if self.last_mode == "ollama":
            from slrag.llm.ollama_client import OllamaClient

            self._client = OllamaClient(config.ollama_url, config.model, timeout_s=config.timeout_s, temperature=config.temperature)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for Qwen2.5 BPE tokenizer."""
        # Standard Qwen / BPE heuristic: word tokens + punctuation splits
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return max(1, len(tokens))

    def calculate_cost(self, tokens_in: int, tokens_out: int) -> float:
        """Calculate generation cost based on pricing."""
        cost_in = (tokens_in / 1000.0) * self.config.cost_per_1k_input
        cost_out = (tokens_out / 1000.0) * self.config.cost_per_1k_output
        return round(cost_in + cost_out, 6)

    def format_qwen_prompt(self, system_prompt: str, user_query: str, context_chunks: List[str]) -> str:
        """Format prompt using Qwen2.5 ChatML template."""
        context_str = "\n\n".join(context_chunks)
        return (
            f"<|im_start|>system\n{system_prompt}\n"
            f"Context:\n{context_str}<|im_end|>\n"
            f"<|im_start|>user\n{user_query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    async def generate_grounded_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> LLMGenerationResult:
        """Generate response and extract atomic claims with citations."""
        return self.generate_grounded_response_sync(query, retrieved_chunks, system_prompt)

    def generate_grounded_response_sync(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> LLMGenerationResult:
        """Synchronous body of generate_grounded_response (usable from inside a running event loop)."""
        if not system_prompt:
            system_prompt = (
                "You are a factual assistant. Answer the user query using only the provided context. "
                "Every factual sentence must be followed by citation in format [DOC_ID§Section]."
            )
        if self._client is not None:
            try:
                return self._generate_with_ollama(query, retrieved_chunks, system_prompt)
            except Exception as exc:  # LLMUnavailableError or an unusable answer: never fail the turn
                logger.warning("Ollama drafting failed (%s); using the extractive drafter for this call", exc)
                self.last_mode = "heuristic_fallback"
        else:
            self.last_mode = "heuristic"

        context_texts = [
            f"[{c['chunk_id']}]: {c['text']}" for c in retrieved_chunks
        ]
        prompt = self.format_qwen_prompt(system_prompt, query, context_texts)
        tokens_in = self.estimate_tokens(prompt)

        # Generate grounded claims from retrieved context
        claims: List[Tuple[str, List[str]]] = []
        answer_sentences: List[str] = []

        for chunk in retrieved_chunks:
            chunk_id = chunk["chunk_id"]
            text = chunk["text"]
            # Extract key informative sentences from the chunk
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            for sent in sentences[:2]:
                claim_text = sent
                claims.append((claim_text, [chunk_id]))
                # Citation goes before the sentence-final punctuation so sentence splitting keeps
                # each citation with its own claim ("Claim text [cid]." not "Claim text. [cid]").
                body = claim_text.rstrip()
                end = body[-1] if body[-1:] in (".", "!", "?") else "."
                answer_sentences.append(f"{body.rstrip('.!?')} [{chunk_id}]{end}")

        full_answer_text = " ".join(answer_sentences)
        tokens_out = self.estimate_tokens(full_answer_text)
        cost = self.calculate_cost(tokens_in, tokens_out)

        return LLMGenerationResult(
            text=full_answer_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            model_name=self.config.model_name,
            raw_claims=claims,
        )

    def _generate_with_ollama(self, query: str, retrieved_chunks: List[Dict[str, Any]], system_prompt: str) -> LLMGenerationResult:
        """One chat call; every sentence of the answer becomes a claim with the [chunk_id] cites it carries.
        Cites are not filtered here: the fail-closed verifier rejects any id outside the evidence."""
        context = "\n\n".join(f"[{c['chunk_id']}]: {c['text']}" for c in retrieved_chunks)
        messages = [
            {"role": "system", "content": f"{system_prompt} Cite exactly one chunk id per bracket, e.g. [doc§section].\nContext:\n{context}"},
            {"role": "user", "content": query},
        ]
        result = self._client.chat_sync(messages)
        text = result["text"].strip()
        claims: List[Tuple[str, List[str]]] = []
        for sentence in (s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()):
            cites = [p.strip() for group in CITE_GROUP_RE.findall(sentence) for p in group.split(",") if p.strip()]
            body = CITE_GROUP_RE.sub("", sentence).strip()
            if body:
                claims.append((re.sub(r"\s+([.!?,;:])", r"\1", body), cites))
        if not claims:
            raise ValueError("empty answer from Ollama")
        tokens_in = result["tokens_in"] or self.estimate_tokens(context + query)
        tokens_out = result["tokens_out"] or self.estimate_tokens(text)
        self.last_mode = "ollama"
        return LLMGenerationResult(
            text=text, tokens_in=tokens_in, tokens_out=tokens_out, cost=self.calculate_cost(tokens_in, tokens_out),
            model_name=self.config.model, raw_claims=claims,
        )
