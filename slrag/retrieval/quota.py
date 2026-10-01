"""Evidence quotas across the sub-queries of one turn."""
from __future__ import annotations

from typing import Any, Dict, List

from slrag.retrieval.cache import chunk_id_of


def apply_quota(results: Dict[str, List[Any]], per_sub: int = 4, global_cap: int = 12) -> Dict[str, List[Any]]:
    """Keep at most `per_sub` chunks per sub-query and `global_cap` in total.

    Chunks are deduplicated within a sub-query only, since one chunk may support several
    sub-intents. When the global cap binds, selection is round-robin by rank across
    sub-queries (in insertion order), so a dominant sub-query cannot starve the others.
    """
    per_sub_lists: Dict[str, List[Any]] = {}
    for sub_id, chunks in results.items():
        seen = set()
        kept: List[Any] = []
        for i, c in enumerate(chunks):
            cid = chunk_id_of(c, f"{sub_id}#{i}")
            if cid in seen:
                continue
            seen.add(cid)
            kept.append(c)
            if len(kept) >= per_sub:
                break
        per_sub_lists[sub_id] = kept

    selected: Dict[str, List[Any]] = {sub_id: [] for sub_id in per_sub_lists}
    total = 0
    for rank in range(per_sub):
        for sub_id, kept in per_sub_lists.items():
            if total >= global_cap:
                return selected
            if rank < len(kept):
                selected[sub_id].append(kept[rank])
                total += 1
    return selected
