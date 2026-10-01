"""Cascade retrieval controller: T0 presentation rules -> T2 LLM (ambiguous band) -> T1 slots/drift."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from slrag.config import ControllerConfig, DEFAULT_CONFIG
from slrag.contracts.events import Claim
from slrag.control.query_builder import build_query
from slrag.control.t0_rules import ledger_vocabulary, t0_score
from slrag.control.t1_slots import t1_decide
from slrag.control.t2_llm import T2Classifier
from slrag.nlp.lemmas import CorpusVocab, anchors, content_tokens, extract_slots, slot_conflict
from slrag.plan.multi_intent_gate import multi_intent_gate
from slrag.retrieval.cache import default_embed


class Decision(str, Enum):
    WAIT = "WAIT"
    PROVISIONAL = "PROVISIONAL"
    COMMIT = "COMMIT"
    SUPPRESS = "SUPPRESS"


class ControllerResult(BaseModel):
    decision: Decision
    tier: Literal["T0", "T1", "T2"]
    query: Optional[str] = None  # Q_t when PROVISIONAL / COMMIT
    reason: str
    commit_hint: Dict[str, Any] = Field(default_factory=lambda: {"multi_intent": False})
    payload: Dict[str, Any] = Field(default_factory=dict)
    # Multi-intent turns: a clause ended on this chunk, so its sub-query can be retrieved now
    # (dispatched by StreamingTurn; independent of `decision`).
    clause_end: bool = False


@dataclass
class TurnState:
    turn_id: str = ""
    buffer: str = ""
    chunks: List[str] = field(default_factory=list)
    last_dispatched_q: Optional[str] = None
    last_dispatched_emb: Optional[np.ndarray] = None
    last_dispatched_slots: Optional[Dict[str, str]] = None
    last_q: Optional[str] = None
    last_q_emb: Optional[np.ndarray] = None
    n_provisional: int = 0
    n_commit: int = 0
    stable_count: int = 0
    committed_q: Optional[str] = None
    epoch: int = 0
    # Per-clause retrieval: set (and kept) when the multi-intent gate fires, before the planner runs.
    is_multi_intent: bool = False
    mi_saw_content: bool = False
    mi_quiet_chunks: int = 0  # consecutive chunks without new content words after a content-word chunk
    # Phase 8: commit-stage pre-draft. pending at COMMIT; committed (stands / refined) or wasted
    # (redone, timed out, cancelled) at utterance_end. predraft_claims are held, never emitted directly.
    predraft_status: Literal["idle", "pending", "committed", "wasted"] = "idle"
    predraft_claims: Optional[List[Claim]] = None


CLAUSE_END_RE = re.compile(r"[?.](?=\s|$)")  # sentence-final ? or . (not the "." in 2.5% or v2.4)


def _ledger_texts(ledger: Any) -> List[str]:
    if ledger is None:
        return []
    claims = ledger.get_verified_claims() if hasattr(ledger, "get_verified_claims") else []
    subs = getattr(ledger, "sub_intents", []) or []
    return [c.text for c in claims] + [s.text for s in subs]


def _has_active_claims(ledger: Any) -> bool:
    return bool(ledger is not None and hasattr(ledger, "get_verified_claims") and ledger.get_verified_claims())


class RetrievalController:
    """Decides, per transcript chunk, whether to wait, retrieve provisionally, commit, or suppress.

    Modes: cascade (T0 -> T2 in the ambiguous band -> T1), rules_only (T2 skipped),
    llm_only (T2 on every chunk), eager (PROVISIONAL on every chunk), batch (always WAIT;
    retrieval happens only at utterance_end).
    """

    def __init__(
        self,
        config: ControllerConfig = DEFAULT_CONFIG.controller,
        vocab: Optional[CorpusVocab] = None,
        t2: Optional[T2Classifier] = None,
        embed: Callable[[str], np.ndarray] = default_embed,
    ):
        self.cfg = config
        self.vocab = vocab
        self.t2 = t2 or T2Classifier(timeout_s=config.t2_timeout_s)
        self.embed = embed

    async def on_chunk(self, turn: TurnState, ledger: Any = None) -> ControllerResult:
        mode = self.cfg.mode
        built = build_query(turn.buffer)
        emb = self.embed(built.query) if built.query else np.zeros(1, dtype=np.float32)
        payload: Dict[str, Any] = {"content_tokens": built.content_count, "dangling": built.dangling, "q_t": built.query}

        if mode == "batch":
            return ControllerResult(decision=Decision.WAIT, tier="T1", reason="batch_mode", payload=payload)

        if mode == "eager":
            if not built.query:
                return ControllerResult(decision=Decision.WAIT, tier="T1", reason="empty_query", payload=payload)
            self._record_dispatch(turn, built.query, emb, Decision.PROVISIONAL)
            turn.last_dispatched_slots = extract_slots(built.query)
            return ControllerResult(decision=Decision.PROVISIONAL, tier="T1", query=built.query, reason="eager", payload=payload)

        # T0 / T2: only meaningful once the session has something to re-present.
        route = "continue"
        if mode == "llm_only":
            route = "ambiguous"
        elif _has_active_claims(ledger):
            t0 = t0_score(
                turn.buffer, ledger_vocabulary(_ledger_texts(ledger)), self.vocab,
                suppress_at=self.cfg.t0_suppress, continue_below=self.cfg.t0_continue,
            )
            payload.update({"t0_score": t0.score, "new_anchors": t0.new_anchors})
            route = t0.route
            if route == "suppress":
                return ControllerResult(decision=Decision.SUPPRESS, tier="T0", reason="presentation_restructure", payload=payload)

        if route == "ambiguous" and mode != "rules_only":
            choice, t2_reason = await self.t2.decide(turn.buffer)
            payload["t2"] = t2_reason
            if choice == "NO_RETRIEVE":
                return ControllerResult(decision=Decision.SUPPRESS, tier="T2", reason="presentation_restructure", payload=payload)

        # T1
        anchor_list = anchors(turn.buffer, self.vocab)
        self._update_stability(turn, built.query, emb, built.dangling)
        gate_fired = multi_intent_gate(turn.buffer).run_planner
        if gate_fired:
            turn.is_multi_intent = True
        clause_end = self._clause_end(turn)
        boundary = not built.dangling and len(anchor_list) >= 2 and gate_fired
        slots = extract_slots(built.query)
        slots_changed = turn.last_dispatched_slots is not None and slot_conflict(slots, turn.last_dispatched_slots) is not None
        t1 = t1_decide(turn, built, anchor_list, emb, turn.buffer, self.cfg, boundary, slots_changed)
        payload.update({"anchors": anchor_list, "drift_cos": t1.drift_cos, "stable_count": turn.stable_count})
        if clause_end:
            payload["clause_end"] = True

        decision = Decision(t1.decision)
        if decision in (Decision.PROVISIONAL, Decision.COMMIT):
            self._record_dispatch(turn, built.query, emb, decision)
            turn.last_dispatched_slots = slots
        return ControllerResult(
            decision=decision,
            tier="T1",
            query=built.query if decision in (Decision.PROVISIONAL, Decision.COMMIT) else None,
            reason=t1.reason,
            commit_hint={"multi_intent": t1.multi_intent},
            payload=payload,
            clause_end=clause_end,
        )

    def _clause_end(self, turn: TurnState) -> bool:
        """Clause-end signal on a multi-intent turn: the latest chunk contains a sentence-final ? or ., or
        two consecutive chunks brought no new content words after a content-word chunk."""
        chunk = turn.chunks[-1] if turn.chunks else ""
        if content_tokens(chunk):
            turn.mi_saw_content, turn.mi_quiet_chunks = True, 0
        elif turn.mi_saw_content:
            turn.mi_quiet_chunks += 1
        if not (self.cfg.per_clause_retrieval and turn.is_multi_intent):
            return False
        if CLAUSE_END_RE.search(chunk):
            return True
        if turn.mi_quiet_chunks >= 2:
            turn.mi_quiet_chunks = 0
            return True
        return False

    def _update_stability(self, turn: TurnState, query: str, emb: np.ndarray, dangling: bool) -> None:
        """Count consecutive chunks (after a PROVISIONAL) whose Q_t barely moved: cos >= stable_cos, no dangling tail."""
        if turn.last_q_emb is not None and turn.n_provisional > 0 and emb.shape == turn.last_q_emb.shape:
            stable = float(np.dot(emb, turn.last_q_emb)) >= self.cfg.stable_cos and not dangling
            turn.stable_count = turn.stable_count + 1 if stable else 0
        turn.last_q, turn.last_q_emb = query, emb

    @staticmethod
    def _record_dispatch(turn: TurnState, query: str, emb: np.ndarray, decision: Decision) -> None:
        turn.last_dispatched_q, turn.last_dispatched_emb = query, emb
        if decision == Decision.PROVISIONAL:
            turn.n_provisional += 1
            turn.stable_count = 0
        else:
            turn.n_commit += 1
            turn.committed_q = query
