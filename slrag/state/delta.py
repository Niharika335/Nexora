"""Apply rewriter decisions to the claim ledger: versioning, claim-level diff, preservation (Phase 7).

keep    -> no change.
revise  -> new text/cites, status "revised", last_modified_version n+1, new text_hash,
           history entry {version, action, text, cites} appended (same claim_id).
retract -> status "retracted"; the claim leaves the live answer but stays in the ledger.
add     -> new claim id, created_version n+1.
Claims of unaffected sub-intents are never touched. answer_version is incremented exactly once.

Everything is validated and built before the ledger is touched, then swapped in one step, so
a rejected delta leaves the ledger unchanged. Afterwards the text_hash of every unaffected and
every kept claim is recomputed and compared with its pre-refinement hash (unchanged_hashes_ok).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence

from slrag.contracts.events import Claim, ClaimStatus, compute_text_hash

logger = logging.getLogger(__name__)


class PreservationError(AssertionError):
    """A claim that must be byte-identical after refinement changed: a bug, never expected."""


@dataclass
class Decision:
    claim_id: str
    action: str  # keep | revise | retract
    text: Optional[str] = None
    cites: Optional[List[str]] = None


@dataclass
class NewClaim:
    sub_intent_id: str
    text: str
    cites: List[str]


@dataclass
class DeltaApplyResult:
    from_version: int
    to_version: int
    kept: List[str]
    revised: List[str]
    retracted: List[str]
    added: List[str]
    unaffected: List[str]
    unchanged_hashes_ok: bool
    preserved_checked: int
    preserved_identical: int
    ops: List[Dict[str, Any]]
    rewrite_input_claim_ids: List[str] = field(default_factory=list)

    def transition_payload(self, **extra: Any) -> Dict[str, Any]:
        payload = {
            "from": self.from_version,
            "to": self.to_version,
            "kept": list(self.kept),
            "revised": list(self.revised),
            "retracted": list(self.retracted),
            "added": list(self.added),
            "unchanged_hashes_ok": self.unchanged_hashes_ok,
            "rewrite_input_claim_ids": list(self.rewrite_input_claim_ids),
            "unaffected": list(self.unaffected),
            "preserved_checked": self.preserved_checked,
            "preserved_identical": self.preserved_identical,
        }
        payload.update(extra)
        return payload


def _as_decision(d: Any) -> Decision:
    return d if isinstance(d, Decision) else Decision(d.claim_id, d.action, getattr(d, "text", None), getattr(d, "cites", None))


def _as_new(n: Any) -> NewClaim:
    return n if isinstance(n, NewClaim) else NewClaim(n.sub_intent_id, n.text, list(n.cites))


def apply_delta(
    ledger: Any,
    affected_sub_intents: Sequence[str],
    decisions: Iterable[Any],
    new_claims: Iterable[Any],
    turn_id: str,
    rewrite_input_claim_ids: Optional[List[str]] = None,
    uncertainty: Iterable[Dict[str, Any]] = (),
    strict: bool = False,
) -> DeltaApplyResult:
    affected = set(affected_sub_intents)
    live: List[Claim] = ledger.get_verified_claims()
    by_id = {c.claim_id: c for c in live}
    decs = [_as_decision(d) for d in decisions]
    news = [_as_new(n) for n in new_claims]

    # 1. Validate everything before touching the ledger.
    seen = set()
    for d in decs:
        if d.claim_id not in by_id:
            raise ValueError(f"decision for unknown or inactive claim '{d.claim_id}'")
        if by_id[d.claim_id].sub_intent_id not in affected:
            raise ValueError(f"decision for claim '{d.claim_id}' outside the affected sub-intents")
        if d.claim_id in seen:
            raise ValueError(f"duplicate decision for claim '{d.claim_id}'")
        if d.action not in ("keep", "revise", "retract"):
            raise ValueError(f"unknown action '{d.action}'")
        if d.action == "revise" and not (d.text and d.text.strip()):
            raise ValueError(f"revise decision for '{d.claim_id}' has no text")
        seen.add(d.claim_id)
    for n in news:
        if n.sub_intent_id not in affected:
            raise ValueError(f"new claim for sub-intent '{n.sub_intent_id}' outside the affected sub-intents")
        if not n.cites:
            raise ValueError("new claim without cites")

    pre_hashes = {c.claim_id: c.text_hash for c in live}
    n0 = ledger.answer_version
    n1 = n0 + 1
    decision_of = {d.claim_id: d for d in decs}

    # 2. Build the new live list (unaffected and kept claims are the same objects).
    new_live: List[Claim] = []
    retracted: List[Claim] = []
    kept, revised, retracted_ids, unaffected = [], [], [], []
    ops: List[Dict[str, Any]] = []
    for c in live:
        d = decision_of.get(c.claim_id)
        if c.sub_intent_id not in affected:
            unaffected.append(c.claim_id)
            new_live.append(c)
            ops.append({"op": "keep", "claim_id": c.claim_id})
        elif d is None or d.action == "keep":
            kept.append(c.claim_id)
            new_live.append(c)
            ops.append({"op": "keep", "claim_id": c.claim_id})
        elif d.action == "revise":
            text, cites = d.text.strip(), list(d.cites if d.cites is not None else c.doc_ids)
            new_live.append(c.model_copy(update={
                "text": text, "doc_ids": cites, "status": "revised", "last_modified_version": n1, "turn_id": turn_id,
                "history": list(c.history) + [{"version": n1, "action": "revise", "text": text, "cites": cites}],
            }))
            revised.append(c.claim_id)
            ops.append({"op": "revise", "claim_id": c.claim_id, "text": text, "cites": cites, "prev_text_hash": pre_hashes[c.claim_id]})
        else:  # retract
            retracted.append(c.model_copy(update={
                "status": "retracted", "last_modified_version": n1,
                "history": list(c.history) + [{"version": n1, "action": "retract", "text": c.text, "cites": list(c.doc_ids)}],
            }))
            retracted_ids.append(c.claim_id)
            ops.append({"op": "retract", "claim_id": c.claim_id})

    added: List[str] = []
    for n in news:
        claim = Claim(
            claim_id=ledger.new_claim_id(), text=n.text.strip(), doc_ids=list(n.cites), verification_status=ClaimStatus.VERIFIED,
            turn_id=turn_id, sub_intent_id=n.sub_intent_id, created_version=n1, last_modified_version=n1,
            history=[{"version": n1, "action": "add", "text": n.text.strip(), "cites": list(n.cites)}],
        )
        # Insert after the last live claim of the same sub-intent so the answer keeps sub-intent order.
        pos = max((i for i, c in enumerate(new_live) if c.sub_intent_id == n.sub_intent_id), default=len(new_live) - 1) + 1
        new_live.insert(pos, claim)
        added.append(claim.claim_id)
        ops.append({"op": "add", "claim_id": claim.claim_id, "text": claim.text, "cites": list(claim.doc_ids)})

    # 3. Commit in one step; bump answer_version exactly once.
    ledger.replace_live_claims(new_live, retracted, turn_id)
    ledger.answer_version = n1
    for item in uncertainty:
        ledger.add_uncertainty(**item)

    # 4. Preservation assertion: recompute hashes of every unaffected and every kept claim.
    must_match = unaffected + kept
    identical = 0
    for cid in must_match:
        claim = ledger.get_claim(cid)
        if claim is not None and compute_text_hash(claim.text, claim.doc_ids) == pre_hashes[cid]:
            identical += 1
    ok = identical == len(must_match)
    if not ok:
        message = f"preservation assertion failed: {len(must_match) - identical} unaffected/kept claim(s) changed"
        if strict:
            raise PreservationError(message)
        logger.error(message)

    return DeltaApplyResult(
        from_version=n0, to_version=n1, kept=kept, revised=revised, retracted=retracted_ids, added=added,
        unaffected=unaffected, unchanged_hashes_ok=ok, preserved_checked=len(must_match), preserved_identical=identical,
        ops=ops, rewrite_input_claim_ids=list(rewrite_input_claim_ids or []),
    )
