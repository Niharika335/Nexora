"""Reconcile a commit-stage pre-draft with the final evidence at utterance_end (Phase 8).

Validity is per sub-intent: a pre-drafted sub-intent is usable only if its sub-query still matches
a final sub-query by the cache reuse rule (cos >= reuse_cos, no slot conflict). Final sub-intents
without a valid pre-draft (e.g. a clause that was still being spoken at COMMIT) are drafted from
scratch; unmatched pre-drafts are discarded and their tokens wasted. A slot change anywhere in the
buffer against the committed text (e.g. 30 -> 50 people) invalidates the whole pre-draft.

Turn outcome, over the evidence the matched sub-intents' drafts are built from:
  STANDS   every final sub-intent has a valid pre-draft, every pre-draft chunk is in the final
           evidence and no new chunk arrived: the pre-draft claims are reused as they are.
  REFINED  at least one pre-draft is reused, the matched sub-intents' evidence adds <= max_new
           chunks and <= max_redrafted final sub-intents need a fresh draft: pre-draft claims are
           reused (those citing a chunk that left the evidence are discarded); only the new chunks
           (and the unmatched sub-intents) are drafted and verified.
  REDONE   otherwise, or on a buffer slot conflict, or when the pre-draft was cancelled / timed
           out / failed: the whole pre-draft is discarded and the full pipeline runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from slrag.answer.predraft import PendingDraft, SubDraft
from slrag.contracts.events import Claim
from slrag.nlp.lemmas import extract_slots, slot_conflict
from slrag.retrieval.cache import chunk_id_of, default_embed


class ReconcileOutcome(str, Enum):
    STANDS = "stands"
    REFINED = "refined"
    REDONE = "redone"


@dataclass
class SubReconcile:
    sub_intent_id: str
    pending: Optional[SubDraft]
    final_ids: List[str]
    new_ids: List[str] = field(default_factory=list)
    dropped_ids: List[str] = field(default_factory=list)

    @property
    def new_chunk_set(self) -> set:
        return set(self.new_ids)


@dataclass
class ReconcileDecision:
    outcome: ReconcileOutcome
    reason: str
    subs: Dict[str, SubReconcile] = field(default_factory=dict)
    new_chunk_ids: List[str] = field(default_factory=list)
    dropped_chunk_ids: List[str] = field(default_factory=list)
    unused_predrafts: List[SubDraft] = field(default_factory=list)  # pre-drafted sub-intents matching no final one
    redrafted: List[str] = field(default_factory=list)  # final sub-intents without a valid pre-draft

    @property
    def reuses_predraft(self) -> bool:
        return self.outcome in (ReconcileOutcome.STANDS, ReconcileOutcome.REFINED)


def match_sub_intents(
    final_subs: Sequence[Tuple[str, str]],
    pending: PendingDraft,
    reuse_cos: float = 0.82,
    embed: Callable[[str], np.ndarray] = default_embed,
) -> Tuple[Dict[str, Optional[SubDraft]], Optional[str]]:
    """Map each final sub-intent to the pre-drafted sub-intent it still matches (cache reuse rule).

    Returns (matches, failure_reason); failure_reason is None when every final sub-intent matched."""
    available = list(pending.sub_drafts.values())
    matches: Dict[str, Optional[SubDraft]] = {}
    reason: Optional[str] = None
    for sid, text in final_subs:
        emb = embed(text)
        best, best_cos, conflict = None, -1.0, None
        for sub in available:
            cos = float(np.dot(emb, embed(sub.text)))
            c = slot_conflict(extract_slots(text), extract_slots(sub.text))
            if c is None and cos > best_cos:
                best, best_cos = sub, cos
            elif c is not None:
                conflict = c
        if best is not None and best_cos >= reuse_cos:
            matches[sid] = best
            available.remove(best)
        else:
            matches[sid] = None
            reason = reason or ("slot_conflict" if conflict else "plan_changed")
    if available and reason is None:
        reason = "plan_changed"  # a pre-drafted sub-intent no longer exists in the final plan
    return matches, reason


def decide(
    final_buffer: str,
    final_subs: Sequence[Tuple[str, str]],
    final_evidence: Dict[str, List[Any]],
    pending: Optional[PendingDraft],
    reuse_cos: float = 0.82,
    max_new_chunks: int = 2,
    ready: bool = True,
    max_redrafted: int = 2,
) -> ReconcileDecision:
    if pending is None:
        return ReconcileDecision(ReconcileOutcome.REDONE, "no_predraft")
    if pending.error:
        return ReconcileDecision(ReconcileOutcome.REDONE, "predraft_error")
    if pending.cancelled or not ready:
        return ReconcileDecision(ReconcileOutcome.REDONE, pending.cancel_reason or "timeout")

    if slot_conflict(extract_slots(final_buffer), pending.committed_slots) is not None:
        return ReconcileDecision(ReconcileOutcome.REDONE, "slot_conflict")
    matches, _ = match_sub_intents(final_subs, pending, reuse_cos)
    matched = {id(m) for m in matches.values() if m is not None}
    unused = [sd for sd in pending.sub_drafts.values() if id(sd) not in matched]
    redrafted = [sid for sid, _ in final_subs if matches[sid] is None]

    subs: Dict[str, SubReconcile] = {}
    new_all: List[str] = []
    dropped_all: List[str] = []
    for sid, _ in final_subs:
        sub = matches[sid]
        final_ids = [chunk_id_of(c, str(i)) for i, c in enumerate(final_evidence.get(sid, []))]
        pre_ids = sub.evidence_ids if sub else []
        new = [c for c in final_ids if c not in pre_ids]
        dropped = [c for c in pre_ids if c not in final_ids]
        subs[sid] = SubReconcile(sid, sub, final_ids, new, dropped)
        new_all.extend(c for c in new if c not in new_all)
        dropped_all.extend(c for c in dropped if c not in dropped_all)

    if not matched:
        outcome, reason = ReconcileOutcome.REDONE, "plan_changed"
    elif not new_all and not dropped_all and not redrafted and not unused:
        outcome, reason = ReconcileOutcome.STANDS, "evidence_unchanged"
    elif len(new_all) <= max_new_chunks and len(redrafted) <= max_redrafted:
        outcome = ReconcileOutcome.REFINED
        reason = f"{len(new_all)}_new_chunks" + (f"_{len(redrafted)}_redrafted" if redrafted else "")
    else:
        outcome = ReconcileOutcome.REDONE
        reason = f"{len(new_all)}_new_chunks" if len(new_all) > max_new_chunks else f"{len(redrafted)}_redrafted"
    return ReconcileDecision(outcome, reason, subs, new_all, dropped_all, unused, redrafted)


def kept_claims(sub: SubDraft, dropped_ids: Sequence[str]) -> Tuple[List[Claim], int]:
    """Verified pre-draft claims of a sub-intent that do not cite a chunk that left the evidence."""
    if sub.verified is None:
        return [], 0
    dropped = set(dropped_ids)
    kept = [c for c in sub.verified.claims if not dropped & set(c.doc_ids)]
    return kept, len(sub.verified.claims) - len(kept)
