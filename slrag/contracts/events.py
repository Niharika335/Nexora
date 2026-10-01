"""Frozen Pydantic v2 models for SL-RAG event contracts."""

from datetime import datetime, timezone
from enum import Enum
import hashlib
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, computed_field

from slrag.config import DEFAULT_CONFIG


def current_iso_time() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_event_id() -> str:
    return f"evt_{uuid.uuid4().hex[:12]}"


class BaseEvent(BaseModel):
    """Base event contract with frozen immutability and metadata."""
    model_config = ConfigDict(frozen=True, extra="ignore")

    event_id: str = Field(default_factory=generate_event_id)
    # ISO 8601 wall-clock string; stream events (StreamEvent) use float seconds instead, so a
    # telemetry log mixing both families still validates against the base contract.
    timestamp: Union[str, float] = Field(default_factory=current_iso_time)
    session_id: str = "default_session"
    seq: int = 0
    cfg_hash: str = Field(default_factory=lambda: DEFAULT_CONFIG.cfg_hash)
    llm_mode: str = Field(default_factory=lambda: DEFAULT_CONFIG.llm.backend)  # heuristic | ollama | heuristic_fallback
    event_type: str = "base_event"


class TranscriptChunk(BaseEvent):
    """Event representing an incremental speech-to-text transcript chunk."""
    event_type: Literal["transcript_chunk"] = "transcript_chunk"
    chunk_id: str = Field(default_factory=lambda: f"tc_{uuid.uuid4().hex[:8]}")
    text: str
    is_final: bool = False
    start_ms: int = 0
    end_ms: int = 0
    speaker: str = "user"
    turn_id: str = ""
    ts_s: Optional[float] = None
    buffer_text: str = ""  # merged utterance buffer after this chunk
    merge_mode: str = ""  # "cumulative" | "delta"


class UtteranceEnd(BaseEvent):
    """Event indicating end of user speech utterance."""
    event_type: Literal["utterance_end"] = "utterance_end"
    utterance_id: str = Field(default_factory=lambda: f"utt_{uuid.uuid4().hex[:8]}")
    final_text: str
    duration_ms: int = 0
    turn_id: str = Field(default_factory=lambda: f"turn_{uuid.uuid4().hex[:8]}")
    ts_s: Optional[float] = None


class SessionEnd(BaseEvent):
    """Event closing a session; the registry drops its state immediately."""
    event_type: Literal["session_end"] = "session_end"


class RetrievalItem(BaseModel):
    model_config = ConfigDict(frozen=True)
    chunk_id: str
    score: float
    rank: int
    source_scores: Dict[str, Any] = Field(default_factory=dict)
    section_title: str = ""
    doc_id: str = ""


class RetrievalEvent(BaseEvent):
    """Event emitted during retrieval query execution."""
    event_type: Literal["retrieval"] = "retrieval"
    turn_id: str
    query: str
    mode: str
    top_k: int
    results: List[RetrievalItem]
    latency_ms: float


class SufficiencyCheckEvent(BaseEvent):
    """Event emitted during sufficiency gate evaluation."""
    event_type: Literal["sufficiency_check"] = "sufficiency_check"
    turn_id: str
    dense_top1_score: float
    coverage_score: float
    passed: bool
    reason: str
    thresholds: Dict[str, float] = Field(default_factory=dict)  # {dense_top1, coverage} the scores were gated on


class ClaimStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


def compute_text_hash(text: str, cites: List[str]) -> str:
    return hashlib.sha256((text + "|" + "|".join(sorted(cites))).encode("utf-8")).hexdigest()


class Claim(BaseModel):
    model_config = ConfigDict(frozen=True)
    claim_id: str = Field(default_factory=lambda: f"clm_{uuid.uuid4().hex[:8]}")
    text: str
    doc_ids: List[str] = Field(default_factory=list)  # Enum-constrained Doc IDs
    # Fail-closed verifier outcome (pending | verified | rejected | superseded).
    verification_status: ClaimStatus = ClaimStatus.PENDING
    turn_id: str = ""
    sub_intent_id: str = "q1"
    history: List[Dict[str, Any]] = Field(default_factory=list)  # [{version, action, text, cites}]
    # Ledger lifecycle (Phase 7): active | revised | retracted.
    status: Literal["active", "revised", "retracted"] = "active"
    created_version: int = 0
    last_modified_version: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def text_hash(self) -> str:
        """sha256(text | sorted cites); derived so it can never drift from the claim text."""
        return compute_text_hash(self.text, self.doc_ids)


class SubIntent(BaseModel):
    """One sub-question of a turn, with its sufficiency gate scores."""
    id: str
    text: str
    status: Literal["answerable", "insufficient", "pending"] = "pending"
    sufficiency: Dict[str, Any] = Field(default_factory=dict)


class Constraint(BaseModel):
    """A late-arriving user constraint and the sub-intents it affects."""
    constraint_id: str
    text: str
    turn_id: str
    affects: List[str] = Field(default_factory=list)


class CheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str  # lexical, semantic, coreference, hallucinated_id, consistency
    passed: bool
    score: float = 1.0
    details: str = ""


class ClaimVerificationEvent(BaseEvent):
    """Event emitted when a claim undergoes 5-check fail-closed verification."""
    event_type: Literal["claim_verification"] = "claim_verification"
    turn_id: str
    claim_id: str
    claim_text: str
    doc_ids: List[str]
    checks: Dict[str, CheckResult]
    passed: bool
    rejection_reason: Optional[str] = None


class Ledger(BaseEvent):
    """Event snapshot of the verified claim ledger."""
    event_type: Literal["ledger_update"] = "ledger_update"
    ledger_version: int = 1
    answer_version: int = 0
    verified_claims: List[Claim] = Field(default_factory=list)
    sub_intents: List[SubIntent] = Field(default_factory=list)
    constraints: List[Constraint] = Field(default_factory=list)
    uncertainty: List[Dict[str, Any]] = Field(default_factory=list)
    active_turn_id: str = ""


class AnswerChunk(BaseEvent):
    """One verified claim streamed to the client."""
    event_type: Literal["answer_chunk"] = "answer_chunk"
    turn_id: str
    answer_version: int
    claim_id: str
    text: str
    cites: List[str] = Field(default_factory=list)
    first_token: bool = False
    ts_s: Optional[float] = None


class TurnOutput(BaseModel):
    """PDF-shaped per-turn output record."""
    retrieval_events: List[Dict[str, Any]] = Field(default_factory=list)
    sub_queries: List[str] = Field(default_factory=list)
    answer: str = ""
    citations: List[str] = Field(default_factory=list)
    uncertainty: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class AnswerDelta(BaseEvent):
    """Event streaming generated answer text delta to client."""
    event_type: Literal["answer_delta"] = "answer_delta"
    delta_id: str = Field(default_factory=lambda: f"dlta_{uuid.uuid4().hex[:8]}")
    turn_id: str
    text_delta: str
    is_final: bool = False
    # Phase 7 claim-level diff: ops are keep {claim_id} | revise {claim_id, text, cites, prev_text_hash}
    # | retract {claim_id} | add {claim_id, text, cites}; change_type is initial | refine | add.
    ops: List[Dict[str, Any]] = Field(default_factory=list)
    change_type: Optional[str] = None
    rendered_answer: str = ""
    answer_version: Optional[int] = None
    hashes: Dict[str, str] = Field(default_factory=dict)  # claim_id -> text_hash of every live claim after the ops


class TurnSummary(BaseEvent):
    """Event summarizing the end of a complete interaction turn."""
    event_type: Literal["turn_summary"] = "turn_summary"
    turn_id: str
    utterance: str
    claims_count: int
    verified_count: int
    rejected_count: int
    tokens_in: int
    tokens_out: int
    cost: float
    latency_ms: float
    status: str = "completed"
    # Phase 9 counters (the trace UI reads these rather than counting events itself).
    ttft_ms: Optional[float] = None  # utterance_end -> first answer delta
    retrieval_calls: Optional[int] = None
    llm_calls: int = 0
    # Phase 8 speculation accounting (None / 0 when no pre-draft ran).
    reconcile_outcome: Optional[str] = None
    predraft_ready_before_end: Optional[bool] = None
    wasted_tokens: int = 0
    wasted_provisional: int = 0
    discarded_predrafts: int = 0


class MetricsSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    total_events: int
    total_turns: int
    active_sessions: int
    p50_latency_ms: float
    p95_latency_ms: float
    events_by_type: Dict[str, int]

# ---------------------------------------------------------------------------
# Phase 4-6 streaming events. Timestamps are float seconds so the replay harness can
# stamp them from its deterministic virtual clock and compute TTFT arithmetically.
# ---------------------------------------------------------------------------


class StreamEvent(BaseEvent):
    """Base for controller, planner and replay events (float `timestamp`)."""
    timestamp: float = Field(default_factory=time.time)  # type: ignore[assignment]
    turn_id: str = ""


class TurnStartEvent(StreamEvent):
    event_type: Literal["turn_start"] = "turn_start"
    query: str = ""


class UtteranceFinalEvent(StreamEvent):
    """Marks utterance_end inside a replayed turn; TTFT is measured from here."""
    event_type: Literal["utterance_final"] = "utterance_final"
    query: str = ""


class FirstTokenEmissionEvent(StreamEvent):
    event_type: Literal["first_token_emission"] = "first_token_emission"
    token: str = ""


class ControllerDecisionEvent(StreamEvent):
    event_type: Literal["controller_decision"] = "controller_decision"
    decision: str = "WAIT"
    tier: str = "T1"
    reason: str = ""
    query: Optional[str] = None
    epoch: int = 0
    payload: Dict[str, Any] = Field(default_factory=dict)
    chunk_seq: int = 0  # 1-based index of the transcript chunk this decision was made on (per turn)
    chunk_text: str = ""  # that chunk's text


class CascadeTriggerEvent(StreamEvent):
    """Event emitted when the token-level cascade trigger evaluates a token."""
    event_type: Literal["cascade_trigger"] = "cascade_trigger"
    token_index: int = 0
    confidence: float = 0.0
    entropy: float = 0.0
    state: str = "IDLE"
    trigger_type: str = "none"


class SpeculativeRetrievalEvent(StreamEvent):
    event_type: Literal["speculative_retrieval"] = "speculative_retrieval"
    retrieval_id: str = ""
    query: str = ""
    trigger: str = "final"  # provisional | multi_intent | final
    is_early: bool = True
    epoch: int = 0
    source: str = "fresh"  # fresh | cache
    chunks_count: int = 0
    latency_ms: float = 0.0


class SpeculativeCacheEvent(StreamEvent):
    """Evidence cache lookup/store (action: hit | miss | set | invalidate)."""
    event_type: Literal["speculative_cache"] = "speculative_cache"
    action: str = "miss"
    query: str = ""
    chunk_ids: List[str] = Field(default_factory=list)
    cosine: Optional[float] = None
    conflict: Optional[str] = None


class SpeculationOutcomeEvent(StreamEvent):
    """Fate of an early retrieval: used | wasted | invalidated | false_trigger | stale_discard."""
    event_type: Literal["speculation_outcome"] = "speculation_outcome"
    outcome: str = "used"
    reason: str = ""
    retrieval_id: str = ""


class SuppressionEvent(StreamEvent):
    """A presentation-only turn answered from the ledger with zero retrievals."""
    event_type: Literal["suppression"] = "suppression"
    reason: str = "presentation_restructure"
    fallback: bool = False
    bullets: int = 0


class SubIntentDecompositionEvent(StreamEvent):
    event_type: Literal["sub_intent_decomposition"] = "sub_intent_decomposition"
    query: str = ""
    sub_intents: List[Dict[str, Any]] = Field(default_factory=list)
    is_compound: bool = False
    source: str = "single"  # llm | fallback | single
    merged: List[Dict[str, Any]] = Field(default_factory=list)
    dropped: List[Dict[str, Any]] = Field(default_factory=list)
    gate_reasons: List[str] = Field(default_factory=list)


class SubIntentRetrievalEvent(StreamEvent):
    event_type: Literal["sub_intent_retrieval"] = "sub_intent_retrieval"
    sub_intent_id: str = ""
    retrieval_query: str = ""
    stage_idx: int = 0
    is_parallel: bool = False
    source: str = "fresh"
    retrieved_chunk_ids: List[str] = Field(default_factory=list)  # fused ranking before quota
    evidence_chunk_ids: List[str] = Field(default_factory=list)  # after per-sub/global quota
    expected_chunk_ids: List[str] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)  # [{chunk_id, doc_id, section, score}] for the evidence


class SubIntentCompletionEvent(StreamEvent):
    event_type: Literal["sub_intent_completion"] = "sub_intent_completion"
    sub_intent_id: str = ""
    status: str = "completed"  # completed | suppressed | uncertain
    claim_ids: List[str] = Field(default_factory=list)
    is_suppressed: bool = False
    is_uncertain: bool = False
    is_unanswerable_ground_truth: Optional[bool] = None
    reason: str = ""
    dense_top1: Optional[float] = None  # sufficiency gate scores for this sub-intent
    coverage: Optional[float] = None
    thresholds: Dict[str, float] = Field(default_factory=dict)


class ReconciliationEvent(StreamEvent):
    event_type: Literal["reconciliation"] = "reconciliation"
    reconciliation_type: str = "append"  # append | non_destructive_update | restart
    affected_claim_ids: List[str] = Field(default_factory=list)


class MultiIntentResolutionEvent(StreamEvent):
    event_type: Literal["multi_intent_resolution"] = "multi_intent_resolution"
    total_sub_intents: int = 0
    resolved_count: int = 0
    suppressed_count: int = 0
    uncertain_count: int = 0
    resolution_recall: float = 1.0
    gold_sub_intents: Optional[int] = None


class VerificationEvent(StreamEvent):
    event_type: Literal["verification"] = "verification"
    grounded: bool = True
    groundedness_score: float = 1.0
    supported_claims: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    citations: List[Any] = Field(default_factory=list)


class TurnCompleteEvent(StreamEvent):
    event_type: Literal["turn_complete"] = "turn_complete"
    ttft_ms: float = 0.0
    total_latency_ms: float = 0.0
    output_text: str = ""
    retrieval_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    est_cost_usd: float = 0.0
    emitted_cites: int = 0  # citations in the emitted answer
    hallucinated_cites: int = 0  # emitted citations not in the evidence of their sub-intent
    llm_calls: int = 0  # drafter / planner / rewriter calls made for this turn (pre-drafts included)
    # Phase 8 speculation accounting (tokens_in/out and est_cost_usd include all pre-draft tokens).
    reconcile_outcome: Optional[str] = None
    predraft_ready_before_end: Optional[bool] = None
    wasted_tokens: int = 0
    wasted_provisional: int = 0
    discarded_predrafts: int = 0


class PlanCompletedEvent(StreamEvent):
    """Delta planner result for a later turn: payload {relation, affected_sub_intents, delta_queries, fallback}."""
    event_type: Literal["plan_completed"] = "plan_completed"
    payload: Dict[str, Any] = Field(default_factory=dict)


class AnswerVersionTransitionEvent(StreamEvent):
    """One answer_version step: payload {from, to, kept, revised, retracted, added, unchanged_hashes_ok,
    rewrite_input_claim_ids, ...}. `from == to` records a refinement turn that applied no change."""
    event_type: Literal["answer_version_transition"] = "answer_version_transition"
    payload: Dict[str, Any] = Field(default_factory=dict)


class DraftCompletedEvent(StreamEvent):
    """A sub-intent draft finished. speculative=True for commit-stage pre-drafts (held, not emitted)."""
    event_type: Literal["draft_completed"] = "draft_completed"
    sub_intent_id: str = ""
    speculative: bool = True
    ready_ts_s: float = 0.0
    sufficient: bool = True
    claims_count: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    payload: Dict[str, Any] = Field(default_factory=dict)


class ReconcileCompletedEvent(StreamEvent):
    """utterance_end reconcile of a commit-stage pre-draft against the final evidence.

    outcome: stands | refined | redone. stands/refined/redone also count claims by how they were
    produced (reused pre-draft claims / claims drafted for new chunks / claims drafted from scratch).
    wasted_tokens = tokens of pre-draft LLM calls whose output was discarded."""
    event_type: Literal["reconcile_completed"] = "reconcile_completed"
    outcome: str = "stands"
    reason: str = ""
    stands: int = 0
    refined: int = 0
    redone: int = 0
    wasted_tokens: int = 0
    predraft_tokens: int = 0
    new_chunk_ids: List[str] = Field(default_factory=list)
    dropped_chunk_ids: List[str] = Field(default_factory=list)
    discarded_claims: int = 0
    discarded_predrafts: int = 0
    predraft_ready_before_end: bool = False
    predraft_late_ms: float = 0.0
    timed_out: bool = False
    predraft_ready_ts_s: Optional[float] = None
