"""Claim ledger with versioning, sub-intents, constraints, uncertainty items and session persistence."""

import copy
import logging
from typing import Any, Dict, List, Optional
from slrag.contracts.events import Claim, ClaimStatus, Constraint, Ledger, SubIntent
from slrag.telemetry.bus import TelemetryBus, GLOBAL_BUS

logger = logging.getLogger(__name__)


class ClaimLedger:
    """Session-scoped versioned ledger recording all verified claims across turns.

    `version` counts ledger mutations (starts at 1). `answer_version` counts content answers:
    it is bumped only when a turn adds verified claims (or a Phase 7 refinement is applied), so
    presentation-only turns that re-render existing claims leave it unchanged. Each claim carries
    a derived `text_hash`, a `history` of {version, action, text, cites} entries and a
    `status` of active | revised | retracted. Retracted claims leave the active answer
    but stay in the ledger for lineage.
    """

    def __init__(self, session_id: str = "default_session"):
        self.session_id = session_id
        self.version = 1
        self.answer_version = 0
        self._verified_claims: List[Claim] = []  # live claims (active | revised), in answer order
        self._superseded_claims: List[Claim] = []
        self._retracted_claims: List[Claim] = []
        self._claims_by_turn: Dict[str, List[Claim]] = {}
        self._sub_intents: Dict[str, SubIntent] = {}
        self._constraints: List[Constraint] = []
        self._uncertainty: List[Dict[str, Any]] = []
        self._next_claim_no = 1

    # -- reads ---------------------------------------------------------------------

    def get_verified_claims(self) -> List[Claim]:
        """Return all live verified claims (active or revised; retracted claims are excluded)."""
        return list(self._verified_claims)

    def active_claims(self) -> List[Claim]:
        return self.get_verified_claims()

    def claims_for(self, sub_intent_ids: List[str]) -> List[Claim]:
        """Live claims of the given sub-intents, in answer order."""
        wanted = set(sub_intent_ids)
        return [c for c in self._verified_claims if c.sub_intent_id in wanted]

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        return next((c for c in self._verified_claims if c.claim_id == claim_id), None)

    def claims_in_sub_intent_order(self) -> List[Claim]:
        """Live claims grouped by sub-intent registration order (stable within a sub-intent)."""
        order = {sid: i for i, sid in enumerate(self._sub_intents)}
        return sorted(self._verified_claims, key=lambda c: order.get(c.sub_intent_id, len(order)))

    @property
    def retracted_claims(self) -> List[Claim]:
        return list(self._retracted_claims)

    @property
    def sub_intents(self) -> List[SubIntent]:
        return list(self._sub_intents.values())

    @property
    def constraints(self) -> List[Constraint]:
        return list(self._constraints)

    @property
    def uncertainty(self) -> List[Dict[str, Any]]:
        return list(self._uncertainty)

    # -- writes --------------------------------------------------------------------

    def _record(self, claim: Claim, turn_id: str) -> None:
        self._verified_claims.append(claim)
        self._claims_by_turn.setdefault(turn_id, []).append(claim)

    def new_claim_id(self) -> str:
        claim_id = f"c{self._next_claim_no}"
        self._next_claim_no += 1
        return claim_id

    def add_verified_claims(self, claims: List[Claim], turn_id: str) -> int:
        """Append new verified claims, increment ledger version, and record turn history."""
        added = 0
        for c in claims:
            if c.verification_status == ClaimStatus.VERIFIED:
                self._record(c, turn_id)
                added += 1

        if added > 0:
            self.version += 1
            self.answer_version += 1
            logger.debug(f"Ledger version bumped to {self.version} for session {self.session_id} (+{added} claims)")

        return self.version

    def add_sub_intent(self, sub_intent: SubIntent) -> None:
        self._sub_intents[sub_intent.id] = sub_intent

    def add_constraint(self, constraint: Constraint) -> None:
        self._constraints.append(constraint)

    def add_uncertainty(
        self,
        sub_intent_id: str,
        reason: str,
        gate_scores: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Record an uncertainty item (reason: insufficient_evidence | no_verified_claims |
        refinement_insufficient | contradiction | refinement_error). No claim is added."""
        item: Dict[str, Any] = {"sub_intent_id": sub_intent_id, "reason": reason, "gate_scores": gate_scores or {}}
        if text is not None:
            item["text"] = text
        item.update(extra)
        self._uncertainty.append(item)

    def add_or_update_claim(
        self,
        sub_intent_id: str,
        text: str,
        doc_ids: Optional[List[str]] = None,
        is_update: bool = False,
        turn_id: str = "",
        verified: bool = True,
    ) -> str:
        """Add a claim for a sub-intent, or non-destructively revise its current claim.

        An update marks the previous active claim SUPERSEDED (kept for history) and adds the
        new text as a fresh claim whose history links back to it. Returns the new claim_id.
        """
        claim_id = self.new_claim_id()
        status = ClaimStatus.VERIFIED if verified else ClaimStatus.PENDING
        cites = list(doc_ids or [])
        history: List[Dict[str, Any]] = [{"version": self.answer_version + 1, "action": "add", "text": text}]

        if is_update:
            previous = next((c for c in reversed(self._verified_claims) if c.sub_intent_id == sub_intent_id), None)
            if previous is not None:
                self._verified_claims.remove(previous)
                self._superseded_claims.append(previous.model_copy(update={"verification_status": ClaimStatus.SUPERSEDED}))
                history = previous.history + [
                    {"version": self.answer_version + 1, "action": "revise", "from": previous.claim_id, "text": text}
                ]

        claim = Claim(
            claim_id=claim_id,
            text=text,
            doc_ids=cites,
            verification_status=status,
            turn_id=turn_id,
            sub_intent_id=sub_intent_id,
            history=history,
            created_version=self.answer_version + 1,
            last_modified_version=self.answer_version + 1,
        )
        if status == ClaimStatus.VERIFIED:
            self._record(claim, turn_id)
            self.version += 1
        return claim_id

    def bump_answer_version(self) -> int:
        """Mark the end of a content answer that added claims via add_or_update_claim."""
        self.answer_version += 1
        return self.answer_version

    def replace_live_claims(self, claims: List[Claim], retracted: List[Claim], turn_id: str) -> None:
        """Swap in a new live-claim list in one step (used by slrag.state.delta.apply_delta)."""
        known = {c.claim_id for c in self._verified_claims}
        self._verified_claims = list(claims)
        for c in claims:
            if c.claim_id not in known:
                self._claims_by_turn.setdefault(turn_id, []).append(c)
        self._retracted_claims.extend(retracted)
        self.version += 1

    # -- fail-safe -------------------------------------------------------------------

    def checkpoint(self) -> Dict[str, Any]:
        """Snapshot of every mutable field, for all-or-nothing refinement (see restore)."""
        return copy.deepcopy({k: v for k, v in self.__dict__.items()})

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        """Roll the ledger back to a checkpoint taken earlier in this session."""
        self.__dict__.update(copy.deepcopy(checkpoint))

    # -- output ----------------------------------------------------------------------

    def render(self) -> str:
        return " ".join(f"{c.text} [{', '.join(c.doc_ids)}]" for c in self._verified_claims)

    def create_snapshot_event(self, turn_id: str, seq: int = 0) -> Ledger:
        """Generate frozen Ledger event snapshot."""
        return Ledger(
            session_id=self.session_id,
            active_turn_id=turn_id,
            seq=seq,
            ledger_version=self.version,
            answer_version=self.answer_version,
            verified_claims=list(self._verified_claims),
            sub_intents=self.sub_intents,
            constraints=self.constraints,
            uncertainty=self.uncertainty,
        )

    async def emit_snapshot(
        self,
        turn_id: str,
        seq: int = 0,
        bus: Optional[TelemetryBus] = None,
    ) -> Ledger:
        """Emit current ledger snapshot event to the telemetry bus."""
        telemetry_bus = bus or GLOBAL_BUS
        snapshot = self.create_snapshot_event(turn_id, seq)
        await telemetry_bus.emit(snapshot)
        return snapshot
