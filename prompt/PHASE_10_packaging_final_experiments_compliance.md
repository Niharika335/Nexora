# Phase 10 — Packaging, Auto-Replay, Final Experiments & Compliance

**Priority:** MUST HAVE (G1 packaging, compliance audit, final experiments) · HIGH VALUE (demo checker, fallback recording)

## Objective

Turn the working system into the submitted, reproducible artifact: `docker compose up --build` on a clean machine builds everything, starts the services, auto-replays the supplied streams, and writes `summary.json` with no manual step (G1). Then freeze the configuration, run the final experiments on the test split, audit compliance with the hard rules, and produce the remaining deliverables from real measured results.

## Starting State

- Phases 1–9 complete and green. Threshold calibration done on the tune split (Phase 6).
- Late-detail and tail-change scenarios exist. A1–A5 have been run at least once.
- Compose already has `llm` and `app` (Phase 3). The UI is served by the app.

## Implementation Scope

**MUST HAVE**
- Final `docker-compose.yml`, `Dockerfile`, `.dockerignore`, hashed lockfile.
- Model download at build time from pinned revisions, sha256-verified. No network at runtime except the internal `llm` service.
- Startup sequence: build/load index → warm-up requests → auto-replay if `/data/replay/*.jsonl` exists → write `out/summary.json` → keep serving the UI.
- Clean-machine reproducibility test (G1).
- Frozen config and final experiment run on the test split (baseline table, A1–A5).
- Compliance audit script (corpus isolation, no hardcoding, session isolation, parsimony).
- `README.md` with the one-command run.
- Benchmark report with ≥ 3 real analyzed edge-case failures and ≥ 2 ablations.
- Architecture brief (≤ 6 pages), telemetry schema doc.

**HIGH VALUE**
- `scripts/demo_check.py`: run each demo scenario and assert its expected behavior.
- Pre-recorded fallback demo recording, per the demo script.

**OPTIONAL**
- Export of telemetry to OpenTelemetry (explicitly not part of the core).

## Files / Modules

Create/modify:
- `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `scripts/fetch_models.sh`, `requirements.lock`
- `src/slrag/server/app.py` (lifespan: index load, warm-up, auto-replay)
- `src/slrag/cli.py` (`replay --auto`, `warmup`)
- `scripts/audit_compliance.py`, `scripts/demo_check.py`
- `docs/architecture_brief.md`, `docs/benchmark_report.md`, `docs/telemetry_schema.md`, `docs/demo_script.md`, `README.md`
- `tests/test_startup.py`, `tests/test_compliance_audit.py`

## Interfaces & Contracts

**Compose services:**
- `llm`: llama.cpp server, model volume `./models:/models:ro`, healthcheck `GET /health`.
- `app`: `depends_on: llm: {condition: service_healthy}`, volumes `./data/corpus:/data/corpus:ro`, `./data/replay:/data/replay:ro`, `./out:/out`, env `LLM_BASE_URL=http://llm:8080/v1`, `LLM_MODEL`, `OUT_DIR=/out`, `INDEX_DIR=/var/index`.
- `app` runs on an internal network with `llm`. No other outbound route is configured.

**Environment variables:** `LLM_BASE_URL`, `LLM_MODEL`, `OUT_DIR`, `INDEX_DIR`, `SLRAG_AUTO_REPLAY` (default `1`), `SLRAG_REPLAY_SPEED` (default `1`), `JUDGE_*` (offline scripts only, never read by `src/slrag` engine modules).

**Outputs on start:** `/out/telemetry.jsonl`, `/out/results.jsonl`, `/out/summary.json`, `/out/report.md`, `/out/traces/`.

**Compliance audit output:** `out/compliance.json` with a pass/fail per rule and the offending file:line for each failure.

## Implementation Steps

1. **Dockerfile finalization.** Multi-stage build. Install from the hashed lockfile with `--require-hashes`. Download the fastembed and spaCy models in a build stage. Copy only `src/`, `config.yaml`, `scripts/fetch_models.sh`. `.dockerignore` excludes `eval/`, `tests/`, `docs/`, `.git`, `out/`, `models/`. Run as a non-root user.
2. **Model pinning.** `fetch_models.sh` downloads the GGUF from a pinned revision URL, verifies sha256 against a value stored in the repo, fails on mismatch, and is idempotent. Provide it as a compose build step or an init service so `docker compose up --build` needs no manual download.
3. **Lifespan startup** (`app.py`). In order:
   - load or build the index (cached in `/var/index`, a named volume),
   - wait for the LLM health, then issue one warm-up completion and one embedding call so the first turn is not cold,
   - if `SLRAG_AUTO_REPLAY=1` and `/data/replay/*.jsonl` exists, run the replay in `ours` mode at 1×, write results and `summary.json`/`report.md`, then continue serving,
   - emit a `startup` telemetry note with timings.
4. **Clean-machine test (G1).** Create a fresh environment (new VM, CI job or a machine with no Docker cache and no local Python packages), clone the repo, place the corpus and replay files, and run exactly `docker compose up --build`. Success = `out/summary.json` exists with G1 `pass`, no manual intervention, and the UI answers on `:8000`. Record the total time.
5. **Offline behavior.** Run the app container with no default route (internal network only). The replay must still complete, proving no external calls are needed. Add `tests/test_startup.py` that asserts the app makes no outbound HTTP except to `LLM_BASE_URL` (patch `httpx` transport in a test, or inspect the requests log).
6. **Freeze config.** Tag the commit. `config.yaml` values are final, `cfg_hash` is written into `summary.json`, and no threshold is changed after the final run.
7. **Final experiments on the test split** with the frozen config, ≥ 3 runs for TTFT medians:
   - baseline table: Ours vs B0 vs B1 (recall@5/@10, groundedness, TTFT p50/p95, cost per turn, early-retrieval, false-trigger, sub-intent recall, over-fragmentation, suppression P/R, uncertainty P/R, preservation, refinement savings, trace coverage),
   - A1 hybrid vs dense vs BM25,
   - A2 eager vs rules_only vs cascade vs llm_only,
   - A3 off vs retrieval_only vs full,
   - A4 delta vs restart,
   - A5 verifier off vs on.
   Groundedness uses the judge plus the human label sample from Phase 6. Report the judge/human agreement.
8. **Edge-case failure analysis.** From the test-run results, select at least 3 actual failures (do not invent them). For each record scenario ID, expected vs actual behavior, root cause (which component and rule), and the mitigation or why it was left. Likely candidates: pronoun-only late details, paraphrased sub-queries below the cache threshold, verifier negation misses, over-fragmentation on list-like single intents.
9. **Compliance audit** (`audit_compliance.py`). Checks, each with pass/fail:
   - **Corpus isolation:** the app image and code make no requests except to `LLM_BASE_URL`. No web-search packages in the lockfile. Every emitted cite in `results.jsonl` exists in the index.
   - **No hardcoding:** search `src/` and prompts for the PDF's example strings (`Pune`, `Venue A`, `catering`, `reimbursement`, `Doc_12`, `Doc_31`, `Doc_09`) and for long substrings of corpus text (≥ 8 consecutive words). No files from `eval/` inside the built image (`docker run --rm image ls /app` check). No precomputed answers or lookup tables keyed by query.
   - **No precomputation:** the replay reads streams at run time. `summary.json` is produced by the run, never shipped.
   - **Session isolation:** a test that runs two sessions concurrently and asserts no shared ledger, cache or buffer state. No files persisted per user. Sessions are gone after `session_end` or idle TTL.
   - **Parsimony:** dependency list contains no agent frameworks or orchestration libraries. Report LLM calls per turn by turn type from telemetry to justify each component.
   - **Prompts:** few-shot exemplars use the synthetic domain only.
   Fix every finding, then rerun until clean.
10. **Demo checker** (`demo_check.py`, HIGH VALUE). For each scenario in `eval/demo/`, replay at 1× and assert the expected behavior using telemetry: early retrieval before `utterance_end`, N sub-queries, refinement preserved hashes, suppression with 0 retrievals, uncertainty present for the unanswerable case. Fail loudly if any demo scenario deviates. Run it three times to confirm stability.
11. **Documents from real numbers:**
    - `docs/architecture_brief.md` (≤ 6 pages): architecture and data flow, trigger logic, decomposition strategy, provenance and grounding, trade-offs and failure-mode mitigations, the PRISM worklet interface (event contract in and out, protocols `Retriever`, `Embedder`, `LLM`, `Verifier`, `StateStore`).
    - `docs/benchmark_report.md`: tables from `summary.json`, ablations, failure analysis, limits (what we do not claim: absolute latency across hardware, real-dollar cost, generic "zero hallucination", anything about the private held-out set).
    - `docs/telemetry_schema.md`: regenerated from the models.
    - `docs/demo_script.md`: exact 5-minute sequence with the utterances, expected screens and the metric to point at, using entities from the corpus.
    - `README.md`: one-command run, expected outputs, config table, data layout, how to swap the corpus and LLM endpoint.
12. **Fallback recording.** Record the exact demo run (screen capture) after `demo_check.py` passes, as a fallback for live failures. The demo uses a pinned seed and temperature 0.
13. **Final freeze.** Tag `v1.0`. Run the clean-machine test once more from the tag.

## Algorithms / Logic

- **Auto-replay:** in `ours` mode at 1×, in file order, sequentially. It writes `results.jsonl` incrementally, so a crash leaves usable partial output. The final step computes and writes `summary.json` and `report.md`.
- **G1 definition used:** the container launches with one command on a clean machine and the automated replay completes without manual intervention (exit status 0 for the replay task, `summary.json` present).
- **Statistics:** TTFT = median of ≥ 3 runs per scenario, then p50/p95 across scenarios. Groundedness and recall are single-run (temperature 0).
- **Safe-claims rule:** the report includes only claims backed by the test split. Every number cites the `cfg_hash` and the git tag it was produced with.

## Integration

- Packages the outputs of Phases 1–9 with no engine changes. Bug fixes found here go back to the relevant module with tests.
- Reuses the Phase 6 harness for the auto-replay and final experiments, the Phase 9 UI and demo scenarios for the demo check, and the Phase 2 coverage checker for G6.

## Verification

```bash
# fresh clone on a clean machine
git clone <repo> && cd streaming-live-rag && cp -r <supplied>/corpus data/corpus && cp -r <supplied>/replay data/replay
time docker compose up --build            # single command
ls out/summary.json out/report.md out/telemetry.jsonl
jq '.gates' out/summary.json
curl -s localhost:8000/health
python scripts/audit_compliance.py --image slrag-app
python scripts/demo_check.py --runs 3
python -m slrag.cli coverage out/telemetry.jsonl
pytest -q
```

Acceptance checks:
- **G1:** `docker compose up --build` on a machine with no cache ends with `out/summary.json` written and the UI reachable, with no manual step. Record wall-clock time.
- **Offline run:** the same works with the app container on an internal-only network.
- **G2–G6** appear in `summary.json` with values and pass/fail on the test split (G2 ≥ 0.80, G3 ≥ 0.70, G4 ≥ 0.85 and 0 hallucinated IDs, G5 preservation = 1.0, G6 coverage = 1.0). If a gate misses, the report says so with the measured value and root cause instead of hiding it.
- **Compliance audit:** all checks pass. `out/compliance.json` clean.
- **Image contents:** no `eval/`, `tests/` or demo scenarios inside the app image.
- **Demo checker:** passes 3 of 3 runs.
- **Documents:** every number in the brief and report matches `summary.json` (spot-check 10 numbers against the file).
- **Regression:** the full test suite from Phases 1–9 passes on the tagged commit.

## Demo Capability

The complete submission: one command from a clean machine to a running system with a live UI and an automatically produced benchmark summary, a scripted five-minute demo whose every scene is validated by an automated checker, and a fallback recording.

## Definition of Done

- [ ] `docker compose up --build` works on a clean machine and produces `summary.json` unattended (G1).
- [ ] Models pinned by revision and sha256. Lockfile hashed. Image contains no eval data.
- [ ] Config frozen and tagged. Final experiments run on the test split (baseline table, A1–A5).
- [ ] ≥ 3 real edge-case failures analyzed, and ≥ 2 ablations reported (five available).
- [ ] Compliance audit clean (corpus isolation, no hardcoding, no precompute, session isolation, parsimony).
- [ ] Architecture brief (≤ 6 pages), benchmark report, telemetry schema, demo script and README complete and consistent with `summary.json`.
- [ ] Demo checker passes 3 of 3. Fallback recording made.
- [ ] Tagged `v1.0`. Full test suite green.

## Failure / Rollback

- First-run slowness or timeouts: raise the compose healthcheck `start_period`, keep the model download in the build stage, and confirm the warm-up runs before auto-replay.
- Replay fails in the container but not locally: compare paths (`/data/...`), file permissions on mounted volumes, and the `OUT_DIR` write access for the non-root user.
- Audit finds corpus text or example strings in code: remove them from prompts or code, move fixtures to `tests/`, and rerun. Never allowlist the finding.
- A gate misses: keep the honest number and analysis. Do not retune on the test split. If a fix is warranted, tune on the tune split, refreeze, and rerun the whole final experiment.
- Rollback: `SLRAG_AUTO_REPLAY=0` disables auto-replay so the UI and CLI still work. All earlier phase entry points (`slrag ask`, `slrag replay`, `/debug/search`) stay functional.

## Output

- Final Dockerfile, compose file, `.dockerignore`, hashed lockfile, model pinning
- Auto-replay startup and `summary.json` / `report.md`
- Compliance audit script and clean report, demo checker
- Architecture brief, benchmark report with failure analysis, telemetry schema, demo script, README
- Tagged `v1.0` and a fallback demo recording
