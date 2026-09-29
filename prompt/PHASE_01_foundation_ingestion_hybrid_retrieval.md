# Phase 1 — Foundation, Corpus Ingestion & Hybrid Retrieval Core

**Priority:** MUST HAVE

## Objective

Create the runnable repository and the corpus-only retrieval substrate everything else depends on: stable `Doc_ID §Section` chunk IDs, a cached BM25 + dense index, RRF fusion, near-duplicate removal, per-sub-query quotas, and the corpus vocabulary/IDF table. Citations, the verifier, the controller's anchor detection and the sufficiency gate all depend on the IDs and vocabulary built here, so they are fixed now.

## Starting State

- Empty repository.
- The supplied corpus placed under `data/corpus/` (format unknown until audited).
- Python 3.11 and Docker available. No LLM needed in this phase.

## Implementation Scope

**MUST HAVE**
- Repo skeleton, `pyproject.toml`, hashed lockfile, `ruff`, `pytest`.
- `config.yaml` and `slrag.config` with `cfg_hash`.
- Corpus loaders: md, txt, json/jsonl, html, pdf.
- Section-aware chunker with stable IDs.
- Index builder cached under a corpus hash.
- BM25 (`bm25s`) and dense (`fastembed` + `bge-small-en-v1.5`, exact numpy matmul) retrieval.
- RRF fusion, dedupe, quota function.
- Corpus vocabulary and IDF table (`vocab.py`), used later by T1 anchors and sufficiency coverage.
- `slrag` CLI: `audit`, `index`, `search`.
- App `Dockerfile` that builds and runs the CLI.

**HIGH VALUE**
- Corpus audit report (doc count, chunk-length histogram, sections per doc, format problems).
- Retrieval probe file (`eval/probes/`) and a probe-scoring command.

**OPTIONAL**
- Table-aware chunking for md/html tables.

## Files / Modules

Create:
- `pyproject.toml`, `requirements.lock`, `config.yaml`, `Dockerfile`, `.dockerignore`, `README.md` (stub)
- `src/slrag/config.py`, `src/slrag/cli.py`, `src/slrag/schemas.py` (only `Chunk`, `Evidence` now)
- `src/slrag/ingest/loader.py`, `chunker.py`, `index.py`, `vocab.py`
- `src/slrag/retrieval/bm25.py`, `dense.py`, `fusion.py`, `retriever.py`
- `src/slrag/nlp/lemmas.py` (spaCy `en_core_web_sm` loader, lemma/stopword helpers)
- `scripts/audit_corpus.py`
- `tests/test_chunker.py`, `test_fusion.py`, `test_index_cache.py`, `test_retriever.py`, `tests/fixtures/mini_corpus/`
- `eval/probes/retrieval_probes.jsonl` (team-authored; excluded from the image)

## Interfaces & Contracts

```python
@dataclass(frozen=True)
class Chunk:
    chunk_id: str      # "Doc_12§2#0"
    cite: str          # "Doc_12 §2"
    doc_id: str
    section: str       # "2"
    text: str
    n_tokens: int

class Evidence(BaseModel):          # matches the frozen Evidence object
    chunk_id: str; cite: str; doc_id: str; section: str; text: str
    bm25_rank: int | None; dense_rank: int | None
    rrf_score: float; dense_cos: float
    retrieval_id: str | None = None; sub_query_ids: list[str] = []

class Retriever(Protocol):
    def search(self, query: str, *, bm25_k=30, dense_k=30, fused_k=10) -> list[Evidence]: ...
    def embed_query(self, text: str) -> np.ndarray: ...     # normalized, bge query prefix
    def embed_texts(self, texts: list[str]) -> np.ndarray: ...
    def chunk(self, chunk_id: str) -> Chunk: ...
    def vocab(self) -> CorpusVocab: ...                     # idf(lemma), has(lemma)

def apply_quota(per_subquery: dict[str, list[Evidence]], per: int = 4, cap: int = 12) -> dict[str, list[Evidence]]
```

`config.yaml` uses the defaults from `IMPLEMENTATION_ROADMAP.md` §4. `cfg_hash` is the first 12 hex chars of sha256 over the canonical JSON of the resolved config.

## Implementation Steps

1. **Skeleton.** Create the package layout, `pyproject.toml` (Python 3.11, deps: fastapi, uvicorn, pydantic>=2, httpx, orjson, fastembed, bm25s, PyStemmer, numpy, spacy, pyyaml, pymupdf or pypdf, beautifulsoup4, pytest, ruff). Generate a hashed lockfile (`pip-compile --generate-hashes`). Pin `en_core_web_sm` by URL and hash.
2. **Config.** Implement `config.py` (dataclass or pydantic model of the schema in the roadmap). Validate unknown keys. Expose `cfg_hash`.
3. **Corpus audit first.** Write `scripts/audit_corpus.py`. Run it on the real corpus and fix loaders based on what it reports before writing the chunker. Record format facts in `README.md`.
4. **Loaders.** `loader.py` yields `RawDoc(doc_id, source_path, sections=[(marker, title, text)])`.
   - md: split on `#` headings.
   - html: split on `h1`–`h6`, strip scripts/styles.
   - txt: split on blank-line-delimited headings if present, else no sections.
   - json/jsonl: read `doc_id`/`id`, `text`/`content`/`body`, and optional `sections`.
   - pdf: extract text per page, detect headings by font size, else no sections.
   - `doc_id` = the `doc_id` field, else the filename stem.
   - Fail loudly on unreadable files, never silently skip.
5. **Section markers.** If a heading starts with a number (`2.`, `2.1`, `Section 2`) use it as the marker. Otherwise use the 1-based heading order. Docs with no headings get sequential §n markers, one per ~200-token window.
6. **Chunker.** `chunker.py`, target 150–250 tokens, overlap 30, using the bge tokenizer for token counts. Split at sentence boundaries. Chunks stay inside one section. A short section is one chunk. Chunk index `#k` counts within `(doc, section)`. Text to embed and BM25-index = `"{section title}. {chunk text}"`. Text stored and shown to the LLM = the chunk text only.
7. **Vocabulary.** `vocab.py` builds a lemma → document-frequency table over chunk texts (spaCy, no stopwords, alphabetic and numeric). `idf(l) = ln((N+1)/(df+1)) + 1`. Persist next to the index.
8. **Index.** `index.py` builds:
   - `bm25s` index over the analyzer output (lowercase, regex `[a-z0-9]+`, stopword removal, PyStemmer).
   - a normalized float32 dense matrix from `fastembed` in batches of 64.
   - chunk metadata and the vocab.
   Save under `${INDEX_DIR:-.cache/index}/<corpus_hash>/`. `corpus_hash` = sha256 over sorted `(relative_path, file_sha256)` plus the chunker config. If the directory exists, load it and skip building.
9. **Dense search.** `dense.py` embeds queries with the prefix `"Represent this sentence for searching relevant passages: "`, computes `matrix @ q`, and uses `argpartition` for top-k.
10. **RRF.** `fusion.py`: `score(d) = Σ_lists 1/(k + rank_list(d))`, ranks 1-based, `k = 60`. Sort descending, tie-break by higher `dense_cos`, then by `chunk_id`.
11. **Dedupe.** Remove a chunk if its `chunk_id` was already kept, or if its cosine with an already kept chunk is ≥ 0.95 (keep the higher RRF).
12. **Retriever.** `retriever.py` combines steps 9–11: run BM25 and dense in parallel (thread executor), fuse, dedupe, keep `fused_k = 10`.
13. **Quota.** `apply_quota` keeps up to 4 chunks per sub-query in RRF order and at most 12 overall. Sub-queries are served round-robin by rank when the global cap binds. It is implemented and tested now and first used in Phase 5.
14. **CLI.** `slrag audit <dir>`, `slrag index <dir>`, `slrag search "<query>" [--mode hybrid|dense|bm25] [--json]`. The `--mode` flag is needed for ablation A1.
15. **Dockerfile.** Multi-stage, Python 3.11-slim, installs from the lockfile, downloads the fastembed model and spaCy model at build time, runs as non-root, `ENTRYPOINT ["python","-m","slrag.cli"]`.
16. **Probes.** Author 15–20 retrieval probes from the audited corpus: `{query, gold_cites:[...]}`. Add `slrag probe` to compute recall@5/10 per mode.

## Algorithms / Logic

- **BM25:** standard `bm25s` (k1 = 1.5, b = 0.75). Top-30.
- **Dense:** cosine over normalized vectors. Top-30.
- **RRF:** as in step 10. Fused list truncated to 10 after dedupe.
- **Dedupe threshold:** 0.95 cosine between chunk embeddings.
- **Quota:** 4 per sub-query, 12 global.
- **IDF:** as in step 7. `CorpusVocab.has(lemma)` is true if the lemma occurs in the corpus.

## Integration

This phase creates the retriever, vocab and config that every later phase imports. Phase 2 wraps `search` with telemetry. Phase 3 uses `Evidence`, `chunk()`, `vocab().idf`. Phase 4 uses `has()` for anchor detection and `embed_query` for drift and cache.

## Verification

```bash
pip install -r requirements.lock && pytest -q
python -m slrag.cli audit data/corpus             # prints: docs, chunks, tokens/chunk p5/p50/p95, docs without headings
python -m slrag.cli index data/corpus             # first run: builds, prints corpus_hash
python -m slrag.cli index data/corpus             # second run: "loaded from cache" in < 2 s
python -m slrag.cli search "<a question the corpus can answer>" --json
python -m slrag.cli probe eval/probes/retrieval_probes.jsonl --mode hybrid
python -m slrag.cli probe eval/probes/retrieval_probes.jsonl --mode dense
docker build -t slrag-app . && docker run --rm -v $PWD/data:/data slrag-app search "<query>"
```

Tests and acceptance checks:
- `test_chunker`: two builds of `tests/fixtures/mini_corpus/` give identical chunk IDs. IDs are unique. Every chunk except a section's last has 40–300 tokens. Markers follow the numbered/sequential rules.
- `test_fusion`: hand-computed RRF for two rank lists with an overlap, for example items A(1,3), B(2,1), C(3,-). Expected scores `1/61+1/63`, `1/62+1/61`, `1/63` and order B, A, C. Dedupe drops an exact duplicate ID and a 0.97-cosine near-duplicate.
- `test_quota`: 3 sub-queries with 10 results each and cap 12 gives ≤ 4 each and exactly 12 total in round-robin order.
- `test_index_cache`: second load is < 2 s and returns an identical `corpus_hash`. Changing one corpus file changes the hash.
- `test_retriever`: every returned `chunk_id` exists in the corpus, and every `Evidence.text` equals the stored chunk text. No text from outside the corpus can appear.
- Probe check: hybrid recall@10 ≥ dense-only recall@10 on the probe set. If not, inspect the analyzer and the tokenization before proceeding.
- Latency: after warm-up, search p95 < 60 ms on ≤ 5k chunks on the dev machine.

## Demo Capability

A terminal demo of corpus-only hybrid retrieval, showing BM25 rank, dense rank, RRF score and stable `Doc_ID §Section` citations for a query. It is also the base of ablation A1.

## Definition of Done

- [ ] `pytest -q` and `ruff check` pass.
- [ ] The real corpus loads with zero silently skipped files. The audit output is saved.
- [ ] Chunk IDs are stable across rebuilds and unique.
- [ ] Index cache works and is keyed by corpus hash.
- [ ] `search --mode hybrid|dense|bm25` all work. RRF, dedupe and quota tests pass.
- [ ] Vocabulary/IDF table is persisted.
- [ ] `docker build` succeeds and `docker run ... search` works.
- [ ] Probe set exists with hybrid ≥ dense recall@10.

## Failure / Rollback

- Empty or poor results: check the analyzer (stemmer applied to both indexing and querying) and that the query prefix is applied for dense search.
- ID collisions: check the section marker rules on documents with duplicate headings.
- Slow index: reduce batch size or check that fastembed uses ONNX CPU threads.
- If the chunker changes after downstream phases exist, all IDs change. Freeze the chunker before Phase 3. Any later change requires a corpus rebuild and re-labelling of gold sections.

## Output

- Repo skeleton, `pyproject.toml`, `requirements.lock`, `config.yaml`, `Dockerfile`
- Corpus loaders, chunker, cached index, vocabulary/IDF table
- Hybrid retriever with RRF, dedupe and quota
- CLI (`audit`, `index`, `search`, `probe`)
- Audit report, probe set, unit tests
