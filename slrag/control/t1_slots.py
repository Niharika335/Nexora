"""T1: slot sufficiency, drift and stability rules -> WAIT / PROVISIONAL / COMMIT."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from slrag.config import ControllerConfig
from slrag.control.query_builder import BuiltQuery
from slrag.nlp.lemmas import has_head_verb


@dataclass
class T1Decision:
    decision: str  # WAIT | PROVISIONAL | COMMIT
    reason: str
    multi_intent: bool = False
    drift_cos: Optional[float] = None


def slots_sufficient(built: BuiltQuery, anchor_list: List[str], buffer: str, cfg: ControllerConfig) -> Optional[str]:
    """Return a WAIT reason if the buffer is not yet specific enough to retrieve, else None."""
    if built.content_count < cfg.min_content_tokens:
        return "too_few_content_tokens"
    n = len(anchor_list)
    if n < 2 and not (n == 1 and has_head_verb(buffer) and built.content_count >= 6):
        return "slots_insufficient"
    return None


def t1_decide(
    turn: Any,
    built: BuiltQuery,
    anchor_list: List[str],
    emb: np.ndarray,
    buffer: str,
    cfg: ControllerConfig,
    multi_intent_boundary: bool,
    slots_changed: bool = False,
) -> T1Decision:
    """Apply the T1 rules to the current turn state (`turn` is a TurnState).

    New information = drift (cos to the last dispatched Q_t below new_info_cos) or a slot change
    (a quantity/date/entity added or altered), since the cache would reject evidence retrieved
    for the old slot values anyway."""
    wait_reason = slots_sufficient(built, anchor_list, buffer, cfg)
    if wait_reason:
        return T1Decision("WAIT", wait_reason)

    drift_cos = None if turn.last_dispatched_emb is None else float(np.dot(emb, turn.last_dispatched_emb))
    can_commit = turn.n_commit < cfg.max_commit

    # A complete multi-intent boundary commits immediately (the planner splits it at COMMIT).
    if can_commit and multi_intent_boundary:
        return T1Decision("COMMIT", "multi_intent_boundary", multi_intent=True, drift_cos=drift_cos)

    new_info = drift_cos is None or drift_cos < cfg.new_info_cos or slots_changed
    if new_info and turn.n_provisional < cfg.max_provisional:
        if turn.last_dispatched_emb is None:
            reason = "first_slot_ok"
        else:
            reason = "new_information" if drift_cos < cfg.new_info_cos else "slot_change"
        return T1Decision("PROVISIONAL", reason, drift_cos=drift_cos)

    if can_commit and turn.n_provisional > 0 and not built.dangling and turn.stable_count >= cfg.stable_chunks - 1:
        return T1Decision("COMMIT", f"stable_{cfg.stable_chunks}", drift_cos=drift_cos)

    return T1Decision("WAIT", "awaiting_stability" if turn.n_commit < cfg.max_commit else "committed", drift_cos=drift_cos)
