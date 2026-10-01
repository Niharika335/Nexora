"""Presenter: re-renders the session ledger for presentation-only turns, with zero retrieval."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from slrag.contracts.events import Claim
from slrag.nlp.lemmas import NUMBER_WORDS, entities, numbers, split_sentences

RenderFn = Callable[[str, List[Claim], int], Awaitable[Dict[str, Any]]]


@dataclass
class PresentationResult:
    bullets: List[Dict[str, Any]]  # {"text", "claim_ids", "cites"}
    fallback: bool
    reason: str = ""
    retrieval_calls: int = 0
    answer_version: int = 0
    failed_checks: List[str] = field(default_factory=list)

    @property
    def rendered(self) -> str:
        return "\n".join(f"- {b['text']} [{', '.join(b['cites'])}]" for b in self.bullets)


def parse_bullet_count(instruction: str, default: int = 3) -> int:
    """Requested bullet count from digits or number words (up to ten)."""
    low = instruction.lower()
    m = re.search(r"\b(\d{1,2})\s+(?:bullet|point|line|sentence|item)", low)
    if m:
        return max(1, int(m.group(1)))
    for word, value in NUMBER_WORDS.items():
        if value <= 10 and re.search(rf"\b{word}\s+(?:bullet|point|line|sentence|item)", low):
            return value
    return default


def post_check(bullets: List[Dict[str, Any]], claims: List[Claim]) -> List[str]:
    """Output must add no cites, numbers or entities beyond the source claims, and only reference active claims."""
    failures: List[str] = []
    source_cites: Set[str] = {c for claim in claims for c in claim.doc_ids}
    source_text = " ".join(c.text for c in claims)
    active_ids = {c.claim_id for c in claims}
    source_entities = {e.lower() for e in entities(source_text)}
    for b in bullets:
        if not set(b.get("claim_ids", [])) <= active_ids:
            failures.append("unknown_claim_id")
        if not set(b.get("cites", [])) <= source_cites:
            failures.append("unseen_cite")
        if not numbers(b.get("text", "")) <= numbers(source_text):
            failures.append("unseen_number")
        if not {e.lower() for e in entities(b.get("text", ""))} <= source_entities:
            failures.append("unseen_entity")
    return sorted(set(failures))


def deterministic_fallback(claims: List[Claim], n: int) -> List[Dict[str, Any]]:
    """First n active claims, each truncated to its first sentence, keeping their cites."""
    return [
        {"text": split_sentences(c.text)[0], "claim_ids": [c.claim_id], "cites": list(c.doc_ids)}
        for c in claims[:n]
    ]


class Presenter:
    def __init__(self, render_fn: Optional[RenderFn] = None):
        self.render_fn = render_fn

    async def render(self, instruction: str, ledger: Any) -> PresentationResult:
        claims = ledger.get_verified_claims()
        n = parse_bullet_count(instruction)
        answer_version = getattr(ledger, "answer_version", 0)

        if self.render_fn is None:
            return PresentationResult(deterministic_fallback(claims, n), fallback=True, reason="no_llm", answer_version=answer_version)

        try:
            raw = await self.render_fn(instruction, claims, n)
            bullets = [
                {
                    "text": str(b["text"]),
                    "claim_ids": list(b.get("claim_ids", [])),
                    "cites": sorted({c for cid in b.get("claim_ids", []) for cl in claims if cl.claim_id == cid for c in cl.doc_ids}),
                }
                for b in raw.get("bullets", [])
            ][:n]
        except Exception:
            return PresentationResult(deterministic_fallback(claims, n), fallback=True, reason="render_error", answer_version=answer_version)

        failures = post_check(bullets, claims) if bullets else ["empty_output"]
        if failures:
            return PresentationResult(
                deterministic_fallback(claims, n), fallback=True, reason="post_check_failed",
                answer_version=answer_version, failed_checks=failures,
            )
        return PresentationResult(bullets, fallback=False, reason="rendered", answer_version=answer_version)
