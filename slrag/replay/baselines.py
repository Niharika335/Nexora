"""Replay modes (B0, B1, ours) and adapters that plug the Phase 1-3 components into TurnEngine."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from slrag.config import AppConfig, DEFAULT_CONFIG, RRFConfig, SlragConfig
from slrag.contracts.events import Claim, VerificationEvent
from slrag.corpus.loader import CorpusLoader
from slrag.nlp.lemmas import CorpusVocab
from slrag.pipeline.drafter import ClaimDrafter
from slrag.pipeline.ledger import ClaimLedger
from slrag.pipeline.llm import LLMServiceWrapper
from slrag.pipeline.sufficiency import SufficiencyGate
from slrag.pipeline.verifier import FailClosedVerifier
from slrag.retrieval.engine import HybridRetrievalEngine, SearchResult

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPORA = (REPO_ROOT / "data" / "sample_corpus.json", REPO_ROOT / "data" / "nexora_corpus.json")

# B0: dense-only retrieval, no verifier, batch (no early retrieval), no planner.
# B1: hybrid retrieval + verifier, batch (retrieval only at utterance_end), no planner.
# ours: hybrid + verifier + cascade controller (early retrieval, cache, suppression) + multi-intent planner
#       + Phase 7 late-detail refinement (B0/B1 restart on a late detail instead).
MODE_FLAGS: Dict[str, Dict[str, Any]] = {
    "b0": dict(dense_weight=1.0, bm25_weight=0.0, enable_verifier=False, enable_cascade=False,
               enable_multi_intent=False, restart_on_late_detail=True, enable_refinement=False),
    "b1": dict(dense_weight=0.6, bm25_weight=0.4, enable_verifier=True, enable_cascade=False,
               enable_multi_intent=False, restart_on_late_detail=True, enable_refinement=False),
    "ours": dict(dense_weight=0.6, bm25_weight=0.4, enable_verifier=True, enable_cascade=True,
                 enable_multi_intent=True, restart_on_late_detail=False, enable_refinement=True),
}


def mode_config(mode: str, overrides: Optional[Dict[str, Any]] = None) -> SlragConfig:
    """Build the SlragConfig for a mode. Override keys are SlragConfig fields, or dotted
    AppConfig paths such as "controller.new_info_cos" / "sufficiency.coverage"."""
    mode = mode.lower()
    if mode not in MODE_FLAGS:
        raise ValueError(f"Unknown replay mode '{mode}'. Expected one of {sorted(MODE_FLAGS)}.")
    cfg = SlragConfig(temperature=0.0, seed=13, **MODE_FLAGS[mode])
    app = cfg.app
    for key, value in (overrides or {}).items():
        if "." in key:
            section, field_name = key.split(".", 1)
            sub = getattr(app, section)
            app = dataclasses.replace(app, **{section: dataclasses.replace(sub, **{field_name: value})})
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.app = dataclasses.replace(app, rrf=RRFConfig(k=app.rrf.k, weight_bm25=cfg.bm25_weight, weight_dense=cfg.dense_weight))
    return cfg


def build_index(app: AppConfig, corpus_paths: Iterable[Path] = DEFAULT_CORPORA) -> HybridRetrievalEngine:
    docs = []
    for path in corpus_paths:
        if Path(path).exists():
            docs.extend(CorpusLoader.load_file(path))
    engine = HybridRetrievalEngine(app)
    engine.index_documents(docs)
    return engine


class RetrieverAdapter:
    """retrieve(query) -> fused top-k SearchResults (dense-only when the BM25 weight is 0,
    BM25-only when the dense weight is 0)."""

    def __init__(self, engine: HybridRetrievalEngine, cfg: SlragConfig):
        self.engine = engine
        self.mode = "dense" if cfg.bm25_weight == 0 else "bm25" if cfg.dense_weight == 0 else "hybrid"
        self.top_k = cfg.top_k

    def retrieve(self, query: str) -> List[SearchResult]:
        return self.engine.search(query, mode=self.mode, top_k=self.top_k)


@dataclass
class GateResult:
    sufficient: bool
    score: float
    reason: str
    dense_top1: float
    coverage: float


class GateAdapter:
    """SufficiencyGate with an object result; `score` is the query coverage."""

    def __init__(self, app: AppConfig):
        self.gate = SufficiencyGate(app.sufficiency)

    def evaluate(self, query: str, chunks: List[Any]) -> GateResult:
        passed, dense_top1, coverage, reason = self.gate.evaluate(query, chunks)
        return GateResult(passed, coverage, reason, dense_top1, coverage)


class DrafterAdapter:
    """Drafts cited claims ("text [chunk_id]") from at most 4 evidence chunks with the LLM wrapper."""

    def __init__(self, app: AppConfig, max_chunks: int = 4):
        self.llm = LLMServiceWrapper(app.llm)
        self.max_chunks = max_chunks
        self.last_usage: Dict[str, float] = {}

    @property
    def llm_mode(self) -> str:
        return self.llm.last_mode

    def draft(self, query: str, chunks: List[Any], temperature: float = 0.0, seed: int = 13) -> str:
        payload = [{"chunk_id": c.chunk_id, "text": c.text} for c in chunks[: self.max_chunks]]
        result = self.llm.generate_grounded_response_sync(query, payload)
        self.last_usage = {"tokens_in": result.tokens_in, "tokens_out": result.tokens_out, "cost": result.cost}
        return result.text


@dataclass
class VerificationResult:
    grounded: bool
    groundedness_score: float
    verified_text: str
    supported: List[str]
    unsupported: List[str]


def cited(claim: Claim) -> str:
    """ "Claim text [cid]." -- the citation goes before the sentence-final punctuation so that
    re-splitting the text into sentences keeps each citation with its own claim."""
    body = claim.text.rstrip()
    end = body[-1] if body[-1:] in (".", "!", "?") else "."
    return f"{body.rstrip('.!?')} [{', '.join(claim.doc_ids)}]{end}"


class VerifierAdapter:
    """Runs the fail-closed verifier on each cited claim of a draft; only passing claims survive."""

    def __init__(self, app: AppConfig, bus: Any, ledger: Optional[ClaimLedger] = None):
        self.verifier = FailClosedVerifier(app.verifier)
        self.bus = bus
        self.ledger = ledger

    def verify(self, draft: str, chunks: List[Any], turn_id: str = "", publish: bool = True) -> VerificationResult:
        evidence = {c.chunk_id: c for c in chunks}
        claims: List[Claim] = ClaimDrafter(set(evidence)).draft_from_text(draft, turn_id=turn_id)
        previous = self.ledger.get_verified_claims() if self.ledger else []
        supported, unsupported, kept = [], [], []
        for claim in claims:
            passed, _, _ = self.verifier.verify_claim(claim, evidence, set(evidence), previous)
            (supported if passed else unsupported).append(claim.claim_id)
            if passed:
                kept.append(cited(claim))
        score = len(supported) / len(claims) if claims else 0.0
        if publish:  # Phase 8 pre-drafts verify silently; the result is published at reconcile
            self.bus.publish(VerificationEvent(
                turn_id=turn_id, grounded=not unsupported and bool(claims), groundedness_score=score,
                supported_claims=supported, unsupported_claims=unsupported,
            ))
        return VerificationResult(not unsupported and bool(claims), score, " ".join(kept), supported, unsupported)

    def verify_claims(self, claims: List[Claim], chunks: List[Any], turn_id: str = "", previous: Optional[List[Claim]] = None) -> VerificationResult:
        """Verify already-atomic claims (Phase 7 revised / new claims) against `chunks`; `previous`
        are the preserved claims used by the consistency check."""
        evidence = {c.chunk_id: c for c in chunks}
        supported, unsupported, kept = [], [], []
        for claim in claims:
            passed, _, _ = self.verifier.verify_claim(claim, evidence, set(evidence), list(previous or []))
            (supported if passed else unsupported).append(claim.claim_id)
            if passed:
                kept.append(cited(claim))
        score = len(supported) / len(claims) if claims else 0.0
        self.bus.publish(VerificationEvent(
            turn_id=turn_id, grounded=not unsupported and bool(claims), groundedness_score=score,
            supported_claims=supported, unsupported_claims=unsupported,
        ))
        return VerificationResult(not unsupported and bool(claims), score, " ".join(kept), supported, unsupported)


def build_turn_engine(
    mode: str,
    index: HybridRetrievalEngine,
    bus: Any,
    clock: Any = None,
    overrides: Optional[Dict[str, Any]] = None,
    ledger: Optional[ClaimLedger] = None,
    delta_llm: Optional[Any] = None,
    rewrite_llm: Optional[Any] = None,
) -> Any:
    """A TurnEngine wired with real components for one replay session (delta_llm / rewrite_llm
    replace the Phase 7 stand-in backends)."""
    from slrag.pipeline.turn_engine import TurnEngine  # local import: turn_engine imports pipeline modules

    cfg = mode_config(mode, overrides)
    ledger = ledger or ClaimLedger()
    return TurnEngine(
        config=cfg,
        retrieval_engine=RetrieverAdapter(index, cfg),
        llm=None,
        verifier=VerifierAdapter(cfg.app, bus, ledger) if cfg.enable_verifier else None,
        sufficiency=GateAdapter(cfg.app),
        ledger=ledger,
        drafter=DrafterAdapter(cfg.app),
        bus=bus,
        clock=clock,
        vocab=CorpusVocab(index.bm25_index.idf_table),
        delta_llm=delta_llm,
        rewrite_llm=rewrite_llm,
    )
