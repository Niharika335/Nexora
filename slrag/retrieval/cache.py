"""Session evidence cache with the cosine + slot-conflict reuse rule."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from slrag.contracts.events import SpeculativeCacheEvent
from slrag.nlp.lemmas import content_tokens, extract_slots, slot_conflict
from slrag.retrieval.dense import generate_bge_small_embedding


def cache_key_text(query: str) -> str:
    """Content-token form of a query, so filler/wh-word differences do not move the embedding."""
    toks = content_tokens(query)
    return " ".join(toks) if toks else " ".join(query.lower().split())


def default_embed(query: str) -> np.ndarray:
    return generate_bge_small_embedding(cache_key_text(query))


def chunk_id_of(chunk: Any, fallback: str = "") -> str:
    return getattr(chunk, "id", None) or getattr(chunk, "chunk_id", None) or fallback


@dataclass
class CacheEntry:
    retrieval_id: str
    query: str
    emb: np.ndarray
    slots: Dict[str, str]
    evidence: List[Any]
    created_at: float
    used: bool = False
    ready_at: float = 0.0  # when the retrieval that produced this entry completes (virtual/wall seconds)


@dataclass
class CacheHit:
    entry: CacheEntry
    cosine: float


@dataclass
class _Candidate:
    entry: CacheEntry
    cosine: float
    conflict: Optional[str] = field(default=None)


class EvidenceCache:
    """Per-session evidence cache.

    Reuse rule: an entry is reused iff cos(query_emb, entry_emb) >= reuse_cos and the two
    queries have no slot conflict (same slot type, different value, e.g. CARDINAL 30 vs 50).
    Entries expire after ttl_ms (None disables expiry) and are evicted LRU beyond max_size.
    """

    def __init__(
        self,
        ttl_ms: Optional[int] = 5000,
        max_size: int = 128,
        bus: Optional[Any] = None,
        clock: Optional[Any] = None,
        reuse_cos: float = 0.82,
        embed: Callable[[str], np.ndarray] = default_embed,
    ):
        self.ttl_ms = ttl_ms
        self.max_size = max_size
        self.bus = bus
        self.clock = clock
        self.reuse_cos = reuse_cos
        self.embed = embed
        self._entries: "OrderedDict[str, CacheEntry]" = OrderedDict()

    def _now(self) -> float:
        return self.clock.time() if self.clock else time.time()

    def _publish(self, **kw: Any) -> None:
        if self.bus:
            self.bus.publish(SpeculativeCacheEvent(timestamp=self._now(), **kw))

    def lookup(self, query: str, emb: Optional[np.ndarray] = None, slots: Optional[Dict[str, str]] = None, turn_id: str = "") -> Optional[CacheHit]:
        """Return the highest-cosine entry satisfying the reuse rule, else None. Emits hit/miss."""
        self.cleanup_expired()
        emb = self.embed(query) if emb is None else emb
        slots = extract_slots(query) if slots is None else slots

        best: Optional[_Candidate] = None
        for entry in self._entries.values():
            cos = float(np.dot(emb, entry.emb))
            conflict = slot_conflict(slots, entry.slots)
            if best is None or (conflict is None, cos) > (best.conflict is None, best.cosine):
                best = _Candidate(entry, cos, conflict)

        if best is not None and best.cosine >= self.reuse_cos and best.conflict is None:
            self._entries.move_to_end(best.entry.retrieval_id)
            self._publish(
                action="hit", query=query, turn_id=turn_id, cosine=round(best.cosine, 4),
                chunk_ids=[chunk_id_of(c, str(i)) for i, c in enumerate(best.entry.evidence)],
            )
            return CacheHit(best.entry, best.cosine)

        self._publish(
            action="miss", query=query, turn_id=turn_id,
            cosine=None if best is None else round(best.cosine, 4),
            conflict=None if best is None else best.conflict,
        )
        return None

    def peek(self, query: str) -> Optional[List[Any]]:
        """Evidence the reuse rule would return for `query`, without publishing a lookup or touching LRU
        (used by the Phase 8 pre-draft so it does not distort cache-hit metrics)."""
        emb, slots = self.embed(query), extract_slots(query)
        best: Optional[_Candidate] = None
        for entry in self._entries.values():
            cos = float(np.dot(emb, entry.emb))
            if slot_conflict(slots, entry.slots) is None and (best is None or cos > best.cosine):
                best = _Candidate(entry, cos, None)
        return list(best.entry.evidence) if best is not None and best.cosine >= self.reuse_cos else None

    def store(
        self,
        retrieval_id: str,
        query: str,
        evidence: List[Any],
        emb: Optional[np.ndarray] = None,
        slots: Optional[Dict[str, str]] = None,
        turn_id: str = "",
        ready_at: Optional[float] = None,
    ) -> CacheEntry:
        self.cleanup_expired()
        self._entries.pop(retrieval_id, None)
        while len(self._entries) >= self.max_size:
            self._entries.popitem(last=False)
        now = self._now()
        entry = CacheEntry(
            retrieval_id=retrieval_id,
            query=query,
            emb=self.embed(query) if emb is None else emb,
            slots=extract_slots(query) if slots is None else slots,
            evidence=list(evidence),
            created_at=now,
            ready_at=now if ready_at is None else ready_at,
        )
        self._entries[retrieval_id] = entry
        self._publish(
            action="set", query=query, turn_id=turn_id,
            chunk_ids=[chunk_id_of(c, str(i)) for i, c in enumerate(evidence)],
        )
        return entry

    def mark_used(self, retrieval_id: str) -> None:
        if retrieval_id in self._entries:
            self._entries[retrieval_id].used = True

    def is_used(self, retrieval_id: str) -> bool:
        entry = self._entries.get(retrieval_id)
        return bool(entry and entry.used)

    # Simple query-keyed API ---------------------------------------------------

    def get(self, query: str) -> Optional[List[Any]]:
        hit = self.lookup(query)
        if hit is None:
            return None
        hit.entry.used = True
        return hit.entry.evidence

    def put(self, query: str, chunks: List[Any]) -> None:
        self.store(retrieval_id=f"q:{cache_key_text(query)}", query=query, evidence=chunks)

    def invalidate(self, query: Optional[str] = None) -> None:
        if query:
            key = cache_key_text(query)
            for rid in [rid for rid, e in self._entries.items() if cache_key_text(e.query) == key]:
                del self._entries[rid]
        else:
            self._entries.clear()
        self._publish(action="invalidate", query=query or "*")

    def cleanup_expired(self) -> None:
        if self.ttl_ms is None:
            return
        now = self._now()
        expired = [rid for rid, e in self._entries.items() if (now - e.created_at) * 1000.0 > self.ttl_ms]
        for rid in expired:
            del self._entries[rid]

    def __len__(self) -> int:
        return len(self._entries)


# Name used by the Phase 4 tests and TurnEngine.
SpeculativeCache = EvidenceCache
