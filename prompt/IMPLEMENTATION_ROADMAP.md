# Streaming Live RAG — Implementation Roadmap

Source of truth: the **FINAL ARCHITECTURE FREEZE**. Nothing here redesigns it. Phases are ordered by dependency, risk, and what the demo needs.

Priority tags used in every phase file: **MUST HAVE**, **HIGH VALUE**, **OPTIONAL**.

---

## 1. Phase Order

| # | Phase | File | Priority |
|---|---|---|---|
| 1 | Foundation, Corpus Ingestion & Hybrid Retrieval Core | `PHASE_01_foundation_ingestion_hybrid_retrieval.md` | MUST HAVE |
| 2 | Event Contracts, Telemetry Bus & Stream Gateway | `PHASE_02_contracts_telemetry_stream_gateway.md` | MUST HAVE |
| 3 | Grounded Answer Pipeline (LLM, Gate, Drafter, Verifier, Ledger v1) | `PHASE_03_grounded_answer_pipeline.md` | MUST HAVE |
| 4 | Cascade Controller, Early Retrieval, Evidence Cache & Suppression | `PHASE_04_cascade_controller_early_retrieval_suppression.md` | MUST HAVE |
| 5 | Multi-Intent Planner, Parallel Retrieval & Per-Sub-Intent Uncertainty | `PHASE_05_multi_intent_planner_parallel_retrieval.md` | MUST HAVE |
| 6 | Replay Harness, Baselines, Metrics, Dev Set & Calibration | `PHASE_06_replay_harness_baselines_metrics.md` | MUST HAVE |
| 7 | Late-Detail Refinement via Claim-Ledger Delta | `PHASE_07_late_detail_claim_ledger_refinement.md` | MUST HAVE |
| 8 | Commit-Stage Pre-Drafting & Reconcile | `PHASE_08_predraft_reconcile.md` | HIGH VALUE |
| 9 | Trace UI | `PHASE_09_trace_ui.md` | MUST HAVE (core panels) / HIGH VALUE (overlay, metrics tab) |
| 10 | Packaging, Auto-Replay, Final Experiments & Compliance | `PHASE_10_packaging_final_experiments_compliance.md` | MUST HAVE |

---

## 2. Dependency Graph

```
P1 Retrieval core ──► P2 Contracts/Telemetry/Gateway ──► P3 Grounded answer (batch, end-to-end)
                                                              │
                                                              ▼
                                                    P4 Controller + early retrieval
                                                       + cache + suppression
                                                              │
                                                              ▼
                                                    P5 Planner + parallel retrieval
                                                       + per-sub-intent uncertainty
                                                              │
                                     ┌────────────────────────┼─────────────────────────┐
                                     ▼                        ▼                         ▼
                           P6 Replay harness          P7 Late-detail refinement   (P6 needed by P7/P8
                           + baselines + metrics      (needs P3 ledger, P5 flow)   for measurement)
                                     │                        │
                                     └────────────┬───────────┘
                                                  ▼
                                        P8 Pre-draft + reconcile
                                        (needs P5 planner, P7 delta machinery, P6 to measure)
                                                  │
                                                  ▼
                                            P9 Trace UI  (reads telemetry from P2–P8; overlay needs P6 traces)
                                                  │
                                                  ▼
                                   P10 Packaging + final experiments + compliance
```

Hard dependencies:

- P2 needs the P1 retriever, so retrieval telemetry exists from the start.
- P3 needs P1 (evidence, IDF table) and P2 (schemas, telemetry).
- P4 needs the P3 pipeline, ledger and LLM client. The Presenter reads ledger claims.
- P5 needs the P4 controller and cache.
- P6 needs P5, so the harness measures the real multi-intent engine. P6 also needs the baseline flags from P3–P5.
- P7 needs the P3 ledger, the P5 flow, and the P6 harness for A4.
- P8 needs the P5 planner and the P7 delta planner and rewriter. It is measured with the P6 harness.
- P9 needs all telemetry events. The B1 overlay needs the P6 trace export.
- P10 needs everything.

---

## 3. What Each Phase Produces

| # | Produces | Architecture components covered | Working capability afterwards |
|---|---|---|---|
| 1 | Runnable repo, pinned deps, `config.yaml`, index builder, hybrid retriever, `slrag search` CLI, app Dockerfile | Corpus loader/chunker, Hybrid Retriever (BM25 + dense + RRF + dedupe + quota) | Ask a query, get ranked evidence with stable `Doc_ID §Section` IDs from the corpus only |
| 2 | Pydantic contracts, telemetry bus (JSONL + WS + `/metrics`), stream gateway, stream simulator, FastAPI shell | Stream Gateway, Telemetry Bus, event contracts | Stream a transcript in real time, see normalized chunks, retrieval and turn events in JSONL |
| 3 | LLM service in compose, LLM client, Sufficiency Gate, Claim Drafter, Claim Verifier, Ledger v1, Emitter, session manager, batch turn engine | Sufficiency Gate, Drafter, Verifier, Ledger (core), Answer Emitter | End-to-end grounded, cited answer at `utterance_end`, with uncertainty on unanswerable questions (first working prototype) |
| 4 | T0/T1/T2 controller, evidence cache, streaming turn state machine, Presenter | Cascade Controller, Evidence Cache, Presenter | Retrieval starts mid-utterance, presentation-only turns are suppressed with 0 retrievals |
| 5 | Query Planner (LLM + spaCy fallback), parallel per-sub-query retrieval, quotas, per-sub-intent gate/draft/uncertainty | Query Planner (decompose), parallel retrieval, quotas | Multi-intent utterances give one unified answer, one citation set per sub-intent |
| 6 | Replay harness, B0/B1 modes, metrics, gates G1–G6 in `summary.json`, 75 dev scenarios, judge, calibration | Replay Harness, baselines | Reproducible numbers for recall, groundedness, TTFT, cost, G2/G3/G4/G6; A1, A2, A5 |
| 7 | Delta planner, rewriter, ledger diff/versioning, restart comparison, 15 late-detail scenarios | Delta-plan mode of Planner, Ledger delta engine | Late constraints revise only affected claims, v1→v2 with diffs; A4 and G5 |
| 8 | Commit-stage pre-drafting, reconcile, waste accounting | Speculation (pre-draft) and reconcile | Drafts ready before `utterance_end`; measured TTFT gain over B1; A3 |
| 9 | Static trace UI, `/api/*` endpoints, B1 overlay, metrics tab | Trace UI | Full jury-visible demo, driven only by telemetry |
| 10 | Final compose, model pinning, auto-replay, compliance audit, final ablations, brief and failure analysis, demo checker | Docker deployment (G1), all remaining deliverables | One command from a clean machine to `summary.json` + UI |

---

## 4. Shared Conventions (all phases)

Repository package: `src/slrag/`. Python 3.11.

**Chunk identity.**
- `chunk_id` = `Doc_12§2#0`
- `cite` = `Doc_12 §2`
- `doc_id` = the `doc_id` field, or the filename stem.

**Config.** One `config.yaml`, loaded by `slrag.config`, hashed into `cfg_hash`. Defaults:

```yaml
retrieval:   {bm25_k: 30, dense_k: 30, rrf_k: 60, fused_k: 10, dedupe_cos: 0.95, quota_per_subquery: 4, quota_global: 12}
controller:  {mode: cascade, min_content_tokens: 4, new_info_cos: 0.82, stable_cos: 0.88, stable_chunks: 2,
              max_provisional: 2, max_commit: 1, t0_suppress: 0.70, t0_continue: 0.40, t2_timeout_s: 0.4}
cache:       {reuse_cos: 0.82}
planner:     {timeout_s: 1.5, max_sub_queries: 4, merge_cos: 0.92, min_content_tokens: 3}
sufficiency: {dense_top1: 0.55, coverage: 0.50, coverage_top_n: 3}
verifier:    {enabled: true, lexical: 0.60, semantic: 0.70}
draft:       {max_claims: 4, max_words: 40}
speculation: {provisional: true, predraft: true}
llm:         {base_url: "http://llm:8080/v1", model: "qwen2.5-3b-instruct-q4_k_m", temperature: 0, seed: 13, parallel: 3}
cost:        {ref_in_per_mtok: 0.10, ref_out_per_mtok: 0.30}   # documented reference rates, not real dollars
session:     {idle_ttl_s: 1800}
```

**Telemetry event types.** `chunk_received`, `controller_decision`, `retrieval_started`, `retrieval_completed`, `cache_lookup`, `plan_completed`, `sufficiency_check`, `draft_completed`, `verification`, `suppression`, `uncertainty_flag`, `answer_emitted`, `answer_version_transition`, `turn_summary`. Phases add fields and payloads, never new required event types.

**Test rule.** Tests may use fixture strings and a tiny synthetic corpus in `tests/fixtures/`. Product code (`src/`) contains no queries, prompts tied to the benchmark domain, canned answers or precomputed results. Few-shot exemplars use a synthetic unrelated domain.

**Every phase ends runnable.** `pytest -q` is green, `docker build` succeeds, and everything from earlier phases still works.

**Repository conventions.**
- Python 3.11, one `pyproject.toml`, hashed `requirements.lock`.
- `ruff` for lint, `pytest` for tests.
- Each phase adds `tests/test_<area>.py` files. Never remove earlier tests.

---

## 5. Critical Path

The minimum sequence for a **working Theme 4 prototype that satisfies the PDF's five capabilities and the G1–G6 gates**:

```
P1 → P2 → P3 → P4 → P5 → P7 → P6 → P10 (minimal: compose up + auto-replay)
```

- **P1–P3**: corpus-grounded, cited, batch RAG with telemetry (first working prototype).
- **P4**: early retrieval and suppression (G2, Example 3).
- **P5**: multi-intent (G3, Example 1).
- **P7**: refinement (G5, Example 2).
- **P6**: measurement, needed to prove G2–G6 and to calibrate thresholds. Run a first pass of P6 right after P5, then extend it after P7.
- **P10 (minimal)**: G1, one command with auto-replay.

P8 (pre-drafting) and P9 (UI) are not on the critical path for compliance. They are on the critical path for **winning**: P8 delivers the TTFT gain, and P9 makes the architecture visible in the demo. Do both if time allows.

Recommended calendar order: P1, P2, P3, P4, P5, P6 (first pass), P7, P6 (extend to A4), P8, P9, P10.

---

## 6. Cut Order

If time runs short, remove in this order. Each cut leaves the core architecture intact.

1. **P9 polish**: SVG timeline, styling, dark mode. Keep the transcript, controller lane, retrieval events, ledger diff and citation click-through.
2. **B1 overlay in the UI** (P9): replace with a metrics-tab table.
3. **A2 arms `llm_only` and `eager`** (P6): keep `rules_only` vs `cascade`.
4. **A1 `bm25_only` arm** (P6): keep hybrid vs dense-only.
5. **Contradiction guard** (P7): keep preservation and versioning.
6. **T2 LLM controller** (P4): the ambiguous band falls back to continuing into T1. Retrieval suppression still works through T0.
7. **Semantic support check (e)** in the verifier (P3): keep (a)–(d).
8. **Pre-drafting** (P8): keep provisional retrieval and the cache. TTFT gain shrinks, and the report says so.
9. **Human label sample**: reduce 50 to 30 claims.

**Never cut:** the replay harness (P6), citation ID validation and enum-constrained cites (P3), the claim ledger and preservation guarantee (P3/P7), suppression (P4), the planner timeout fallback (P5), `docker compose up` (P10), and the trace coverage check (P2/P10).

---

## 7. Final Repository State

```
streaming-live-rag/
├── README.md                       # one-command run, config, data layout
├── pyproject.toml
├── requirements.lock               # hashed
├── config.yaml
├── Dockerfile
├── docker-compose.yml              # services: llm, app
├── .dockerignore                   # excludes eval/, tests/, docs media
├── data/
│   ├── corpus/                     # supplied corpus (mounted, read-only)
│   └── replay/                     # supplied replay streams (mounted)
├── out/                            # telemetry.jsonl, summary.json, traces/ (mounted)
├── models/                         # fetched at build, sha256-verified (Dockerfile.llm/fetch script)
├── scripts/
│   ├── fetch_models.sh
│   ├── audit_corpus.py
│   ├── validate_scenarios.py
│   ├── calibrate.py
│   ├── audit_compliance.py
│   └── demo_check.py
├── src/slrag/
│   ├── config.py  schemas.py  cli.py
│   ├── ingest/     loader.py chunker.py index.py vocab.py
│   ├── retrieval/  bm25.py dense.py fusion.py retriever.py cache.py
│   ├── stream/     gateway.py simulator.py
│   ├── control/    t0_rules.py t1_slots.py t2_llm.py query_builder.py controller.py
│   ├── plan/       planner.py splitter.py multi_intent_gate.py delta_planner.py
│   ├── llm/        client.py schemas.py prompts.py
│   ├── answer/     gate.py drafter.py verifier.py presenter.py rewriter.py emitter.py
│   ├── state/      session.py ledger.py delta.py
│   ├── engine/     turn_engine.py predraft.py reconcile.py modes.py
│   ├── telemetry/  bus.py events.py metrics.py coverage.py
│   ├── replay/     runner.py metrics.py report.py baselines.py
│   ├── nlp/        lemmas.py numbers.py negation.py
│   └── server/     app.py routes.py static/ (index.html app.js style.css fields.json)
├── eval/                           # NOT in the app image
│   ├── scenarios/ (tune/, test/), split.json, gold/
│   ├── probes/, demo/, judge.py, label_sheet.csv, calibration/
├── tests/                          # unit + integration + fixtures
└── docs/
    ├── architecture_brief.md       # ≤ 6 pages (+ PDF)
    ├── benchmark_report.md         # baseline comparison, A1–A5, ≥3 edge-case failures
    ├── telemetry_schema.md
    └── demo_script.md
```

At the end: `docker compose up --build` on a clean machine builds both images, starts the LLM and app, auto-replays `data/replay/*.jsonl` if present, writes `out/summary.json` and `out/telemetry.jsonl`, and serves the UI on `:8000`.
