#!/usr/bin/env python3
"""Download the dense retrieval model into models/ (run once; the Docker build runs it too).

The engine then loads it with local_files_only=True, so no network call is made at run time.
Usage: python scripts/fetch_models.py
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from slrag.config import DEFAULT_CONFIG  # noqa: E402
from slrag.retrieval.dense import MODEL_DIR  # noqa: E402


def main() -> int:
    from sentence_transformers import SentenceTransformer

    name = DEFAULT_CONFIG.dense.model_name
    model = SentenceTransformer(name, cache_folder=str(MODEL_DIR), device="cpu")
    dim = model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension") else model.get_sentence_embedding_dimension()
    print(f"{name}: {dim}-d, cached in {MODEL_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
