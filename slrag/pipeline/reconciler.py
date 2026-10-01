"""Non-destructive plan reconciler: writes sub-intent answers into the claim ledger."""
from __future__ import annotations

from typing import Any, List, Optional
from slrag.contracts.events import ReconciliationEvent


class PlanReconciler:
    def __init__(self, bus: Optional[Any] = None, clock: Optional[Any] = None):
        self.bus = bus
        self.clock = clock

    def reconcile(
        self,
        ledger: Any,
        sub_intent_id: str,
        claim_text: str,
        is_update: bool = False,
        turn_id: str = "",
        reconciliation_type_override: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        verified: bool = True,
    ) -> str:
        """Append a claim for the sub-intent, or revise its current claim without deleting history."""
        rec_type = reconciliation_type_override or ("non_destructive_update" if is_update else "append")
        if ledger is not None and hasattr(ledger, "add_or_update_claim"):
            claim_id = ledger.add_or_update_claim(
                sub_intent_id, claim_text, doc_ids=doc_ids, is_update=is_update, turn_id=turn_id, verified=verified,
            )
        else:
            claim_id = sub_intent_id

        if self.bus:
            kw = {"turn_id": turn_id, "reconciliation_type": rec_type, "affected_claim_ids": [claim_id]}
            if self.clock:
                kw["timestamp"] = self.clock.time()
            self.bus.publish(ReconciliationEvent(**kw))
        return claim_id
