"""Global configuration loaded from config.yaml, and configuration hashing for SL-RAG."""

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)

CONFIG_ENV_VAR = "SLRAG_CONFIG"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass(frozen=True)
class ChunkerConfig:
    min_tokens: int = 300
    max_tokens: int = 500
    overlap_tokens: int = 50
    approx_chars_per_token: float = 4.0


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25


@dataclass(frozen=True)
class DenseConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    normalize_embeddings: bool = True
    batch_size: int = 32


@dataclass(frozen=True)
class RRFConfig:
    k: int = 60
    weight_bm25: float = 0.5
    weight_dense: float = 0.5


@dataclass(frozen=True)
class SufficiencyConfig:
    dense_top1: float = 0.55
    coverage: float = 0.50
    # An answered sub-intent is flagged uncertain when its query coverage is in [uncertain_low, uncertain_high).
    uncertain_low: float = 0.50
    uncertain_high: float = 0.70


@dataclass(frozen=True)
class VerifierConfig:
    lexical: float = 0.30
    semantic: float = 0.65
    coreference_check_enabled: bool = True
    hallucinated_id_check_enabled: bool = True
    consistency_check_enabled: bool = True


@dataclass(frozen=True)
class LLMConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    cost_per_1k_input: float = 0.00015
    cost_per_1k_output: float = 0.00060
    temperature: float = 0.1
    max_tokens: int = 512
    # heuristic: deterministic stand-ins (extractive drafter, lexical delta planner / rewriter), no model needed.
    # ollama: a local Ollama server drafts answers and makes the delta-planner / rewriter JSON calls.
    backend: str = "heuristic"  # heuristic | ollama
    ollama_url: str = "http://localhost:11434"
    model: str = "qwen2.5:3b"  # Ollama model tag
    timeout_s: float = 10.0


@dataclass(frozen=True)
class TelemetryConfig:
    jsonl_log_path: str = "telemetry.jsonl"
    buffer_size: int = 20000
    drain_timeout_sec: float = 5.0


@dataclass(frozen=True)
class SessionConfig:
    idle_timeout_seconds: float = 1800.0
    cleanup_interval_seconds: float = 60.0


@dataclass(frozen=True)
class CascadeConfig:
    enabled: bool = True
    confidence_threshold: float = 0.72
    entropy_threshold: float = 0.38
    min_token_boundary: int = 4
    cache_ttl_ms: int = 5000
    cache_max_size: int = 128


@dataclass(frozen=True)
class MultiIntentConfig:
    enabled: bool = True
    suppression_threshold: float = 0.45
    enable_parallel_retrieval: bool = True
    restart_on_late_detail: bool = False


@dataclass(frozen=True)
class ControllerConfig:
    """Cascade retrieval controller (T0 rules, T1 slots/drift, T2 LLM)."""
    mode: str = "cascade"  # cascade | rules_only | llm_only | eager | batch
    min_content_tokens: int = 4
    new_info_cos: float = 0.82
    stable_cos: float = 0.88
    stable_chunks: int = 2
    max_provisional: int = 2
    max_commit: int = 1
    t0_suppress: float = 0.70
    t0_continue: float = 0.40
    t2_timeout_s: float = 0.4
    # Multi-intent turns: retrieve each sub-query as soon as its clause ends (before utterance_end),
    # independently of the COMMIT transition.
    per_clause_retrieval: bool = True


@dataclass(frozen=True)
class CacheConfig:
    """Session evidence cache reuse rule: cos >= reuse_cos and no slot conflict."""
    reuse_cos: float = 0.82


@dataclass(frozen=True)
class PlannerConfig:
    timeout_s: float = 1.5
    max_sub_queries: int = 4
    merge_cos: float = 0.92
    min_content_tokens: int = 3
    quota_per_subquery: int = 4
    quota_global: int = 12


@dataclass(frozen=True)
class RefinementConfig:
    """Phase 7 late-detail refinement (claim-ledger delta instead of restart)."""
    enabled: bool = True
    planner_timeout_s: float = 1.5
    rewriter_timeout_s: float = 1.5
    max_delta_queries: int = 2
    max_new_claims: int = 3
    evidence_per_sub_intent: int = 4
    strict_preservation: bool = False  # raise on a failed preservation assertion (tests); log an error otherwise


@dataclass(frozen=True)
class SpeculationConfig:
    """Phase 8 speculation. `mode` selects the A3 arm: off (no early retrieval, no pre-draft),
    retrieval_only (provisional/commit retrieval + cache, drafting at utterance_end) or full
    (retrieval + commit-stage pre-draft + reconcile). `provisional` / `predraft` toggle each part
    independently within the arm."""
    mode: str = "full"  # off | retrieval_only | full
    provisional: bool = True
    predraft: bool = True
    predraft_timeout_s: float = 2.0
    refine_max_new_chunks: int = 2
    draft_concurrency: int = 3

    @property
    def provisional_on(self) -> bool:
        return self.provisional and self.mode in ("retrieval_only", "full")

    @property
    def predraft_on(self) -> bool:
        return self.predraft and self.provisional_on and self.mode == "full"


@dataclass(frozen=True)
class UIConfig:
    """HTTP Basic credentials for /ui/* and /api/* (env SLRAG_UI_USERNAME / SLRAG_UI_PASSWORD override)."""
    username: str = "admin"
    password: str = "slrag"


@dataclass(frozen=True)
class AppConfig:
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    dense: DenseConfig = field(default_factory=DenseConfig)
    rrf: RRFConfig = field(default_factory=RRFConfig)
    sufficiency: SufficiencyConfig = field(default_factory=SufficiencyConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    cascade: CascadeConfig = field(default_factory=CascadeConfig)
    multi_intent: MultiIntentConfig = field(default_factory=MultiIntentConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    speculation: SpeculationConfig = field(default_factory=SpeculationConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    # Top-level `frozen: true` in config.yaml marks the final, measured configuration. It is
    # metadata, not a setting: it is left out of to_dict() and therefore out of cfg_hash.
    frozen: bool = field(default=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("frozen", None)
        return data

    @property
    def cfg_hash(self) -> str:
        """Hash of the effective configuration (the values loaded from config.yaml): any value change changes it."""
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def config_from_dict(data: Dict[str, Any]) -> AppConfig:
    """Build an AppConfig from nested {section: {key: value}}; missing keys keep their defaults.

    Unknown sections or keys raise ValueError so a typo in config.yaml cannot silently fall back to a default.
    """
    sections = {f.name: f for f in fields(AppConfig) if f.name != "frozen"}
    data = dict(data or {})
    kwargs: Dict[str, Any] = {"frozen": bool(data.pop("frozen", False))}
    for section, values in data.items():
        if section not in sections:
            raise ValueError(f"Unknown config section '{section}'. Expected one of {sorted(sections)}.")
        section_cls = sections[section].default_factory  # type: ignore[misc]
        allowed = {f.name for f in fields(section_cls)}
        unknown = set(values or {}) - allowed
        if unknown:
            raise ValueError(f"Unknown key(s) {sorted(unknown)} in config section '{section}'. Allowed: {sorted(allowed)}.")
        kwargs[section] = section_cls(**(values or {}))
    return AppConfig(**kwargs)


def load_config(path: Optional[Union[str, Path]] = None) -> AppConfig:
    """Load config.yaml (explicit path, else $SLRAG_CONFIG, else the repository's config.yaml)."""
    config_path = Path(path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH)
    if not config_path.exists():
        if path or os.environ.get(CONFIG_ENV_VAR):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        logger.warning("config.yaml not found at %s; using built-in defaults.", config_path)
        return AppConfig()
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Deployment-only overrides (Docker Compose points the app at the ollama service).
    for env, key in (("SLRAG_LLM_BACKEND", "backend"), ("SLRAG_OLLAMA_URL", "ollama_url")):
        if os.environ.get(env):
            data.setdefault("llm", {})[key] = os.environ[env]
    return config_from_dict(data)


# Default singleton instance, loaded from config.yaml
DEFAULT_CONFIG = load_config()


@dataclass
class SlragConfig:
    """Mutable run configuration for the replay TurnEngine (ours / B0 / B1 modes).

    Component thresholds (controller, cache, planner, sufficiency, verifier) come from `app`.
    """
    dense_weight: float = 0.6
    bm25_weight: float = 0.4
    top_k: int = 10
    temperature: float = 0.0
    seed: int = 13
    enable_verifier: bool = True
    sufficiency_threshold: float = 0.65

    enable_cascade: bool = True
    cascade_confidence_threshold: float = 0.72
    cascade_entropy_threshold: float = 0.38
    min_token_boundary: int = 4
    speculative_cache_ttl_ms: int = 5000
    speculative_cache_max_size: int = 128

    enable_multi_intent: bool = True
    suppression_threshold: float = 0.45

    restart_on_late_detail: bool = False
    enable_refinement: bool = False  # Phase 7: route later content turns of a session through the delta planner

    app: AppConfig = field(default_factory=lambda: DEFAULT_CONFIG)