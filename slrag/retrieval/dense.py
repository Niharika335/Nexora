"""Dense retrieval: sentence-transformers embeddings (config dense.model_name, all-MiniLM-L6-v2 by
default) with cosine similarity, plus the hashed n-gram vector used for fast lexical similarity.

The model is loaded from models/ with local_files_only=True: no network call at run time.
Download it once with `python scripts/fetch_models.py`.
"""

from functools import lru_cache
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from slrag.config import DenseConfig, DEFAULT_CONFIG
from slrag.corpus.models import Chunk

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("SLRAG_MODEL_DIR") or Path(__file__).resolve().parents[2] / "models")


@lru_cache(maxsize=4)
def load_model(model_name: str) -> Any:
    """The sentence-transformers model, from the local cache only (one instance per process)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject.toml
        raise RuntimeError("sentence-transformers is not installed: pip install -e .") from exc
    try:
        return SentenceTransformer(model_name, cache_folder=str(MODEL_DIR), device="cpu", local_files_only=True)
    except Exception as exc:
        raise RuntimeError(f"Dense model '{model_name}' not found in {MODEL_DIR}: run python scripts/fetch_models.py") from exc


_TEXT_CACHE: Dict[Tuple[str, str], np.ndarray] = {}


def encode(texts: List[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    """Unit-normalized embeddings, one row per text (memoized per process: the same corpus is
    re-indexed by every replay arm and test)."""
    missing = list(dict.fromkeys(t for t in texts if (model_name, t) not in _TEXT_CACHE))
    if missing:
        vecs = load_model(model_name).encode(missing, batch_size=batch_size, normalize_embeddings=True,
                                             convert_to_numpy=True, show_progress_bar=False)
        for text, vec in zip(missing, np.asarray(vecs, dtype=np.float32)):
            _TEXT_CACHE[(model_name, text)] = vec
    return np.stack([_TEXT_CACHE[(model_name, t)] for t in texts]) if texts else np.empty((0, 0), dtype=np.float32)


@lru_cache(maxsize=4096)
def _encode_query(text: str, model_name: str) -> np.ndarray:
    return encode([text], model_name)[0]


def generate_bge_small_embedding(text: str, dim: int = 384) -> np.ndarray:
    """Hashed n-gram vector (word unigrams, bigrams, character trigrams) projected into `dim`
    dimensions and L2-normalized. Not a learned embedding: it measures surface overlap. The
    controller, evidence cache and verifier use it for query-stability and overlap checks; their
    thresholds are calibrated on it. Retrieval uses DenseIndex (sentence-transformers) instead.
    """
    if not text or not text.strip():
        return np.zeros(dim, dtype=np.float32)

    vec = np.zeros(dim, dtype=np.float32)
    words = [w.lower() for w in text.split() if w.strip()]
    if not words:
        return np.zeros(dim, dtype=np.float32)

    # Word unigrams, bigrams, and character trigrams
    features: List[Tuple[str, float]] = []
    for i, w in enumerate(words):
        features.append((w, 1.0))
        if i + 1 < len(words):
            features.append((f"{w}_{words[i+1]}", 1.2))
        for j in range(len(w) - 2):
            features.append((w[j:j+3], 0.5))

    for feat, weight in features:
        # Generate 4 pseudo-random feature indices from sha256 hash
        h = hashlib.sha256(feat.encode("utf-8")).digest()
        for k in range(0, 16, 4):
            val = int.from_bytes(h[k:k+4], "little", signed=True)
            idx = abs(val) % dim
            sign = 1.0 if val >= 0 else -1.0
            vec[idx] += sign * weight

    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec = vec / norm
    return vec


class DenseIndex:
    """Dense embedding index with cosine similarity search and graceful missing handling."""

    def __init__(self, config: DenseConfig = DEFAULT_CONFIG.dense):
        self.config = config
        self.doc_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None  # Shape: (N, 384)
        self.dim = config.embedding_dim

    def fit(self, chunks: List[Chunk]) -> "DenseIndex":
        """Encode every chunk ("section title: text") with the sentence-transformers model."""
        self.doc_ids = [c.chunk_id for c in chunks]
        if not chunks:
            self.embeddings = np.empty((0, self.dim), dtype=np.float32)
            return self
        texts = [f"{chunk.section_title}: {chunk.text}" for chunk in chunks]
        self.embeddings = encode(texts, self.config.model_name, self.config.batch_size)
        self.dim = int(self.embeddings.shape[1])
        return self

    def embed_query(self, query: str) -> np.ndarray:
        """Unit-normalized query embedding (cached per query text)."""
        return _encode_query(query, self.config.model_name)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search dense index using cosine similarity."""
        if self.embeddings is None or len(self.doc_ids) == 0:
            logger.warning("DenseIndex is empty or missing embeddings. Returning empty list.")
            return []

        try:
            q_vec = self.embed_query(query)
            q_norm = np.linalg.norm(q_vec)
            if q_norm < 1e-9:
                return [(doc_id, 0.0) for doc_id in self.doc_ids[:top_k]]

            # Dot product with normalized document vectors
            sims = np.dot(self.embeddings, q_vec)
            # Clip between -1.0 and 1.0
            sims = np.clip(sims, -1.0, 1.0)
            # Map cosine similarity to [0.0, 1.0] scale: (cos + 1) / 2
            scores = (sims + 1.0) / 2.0

            ranked_indices = np.argsort(-scores)
            results = [(self.doc_ids[idx], float(scores[idx])) for idx in ranked_indices[:top_k]]
            return results
        except Exception as e:
            logger.error(f"Error during dense search: {e}", exc_info=True)
            # Graceful handling: fallback to empty/zero
            return [(doc_id, 0.0) for doc_id in self.doc_ids[:top_k]]

    def save(self, path: Path):
        """Save dense index to disk."""
        data = {
            "doc_ids": self.doc_ids,
            "dim": self.dim,
            "model_name": self.config.model_name,
        }
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(data, f)
        if self.embeddings is not None:
            np.save(path.with_suffix(".npy"), self.embeddings)

    @classmethod
    def load(cls, path: Path, config: DenseConfig = DEFAULT_CONFIG.dense) -> "DenseIndex":
        """Load dense index from disk."""
        json_path = path.with_suffix(".json")
        npy_path = path.with_suffix(".npy")

        if not json_path.exists():
            logger.warning(f"Dense index metadata not found: {json_path}")
            return cls(config=config)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("model_name") != config.model_name:
            logger.warning("Dense index %s was built with %s, not %s: rebuild it with `slrag index`.",
                           json_path, data.get("model_name"), config.model_name)
        idx = cls(config=config)
        idx.doc_ids = data.get("doc_ids", [])
        idx.dim = data.get("dim", config.embedding_dim)

        if npy_path.exists():
            idx.embeddings = np.load(npy_path)
        else:
            logger.warning(f"Dense index embeddings not found: {npy_path}. Missing embeddings handled gracefully.")
            idx.embeddings = np.zeros((len(idx.doc_ids), idx.dim), dtype=np.float32)

        return idx
