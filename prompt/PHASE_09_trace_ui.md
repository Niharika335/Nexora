# Phase 9 — Trace UI

**Priority:** MUST HAVE (core panels) · HIGH VALUE (B1 overlay, metrics tab) · OPTIONAL (styling, timeline polish)

## Objective

Make every architectural claim visible to the jury. Build a single static page, served by FastAPI, that renders only the telemetry stream: transcript with per-chunk controller decisions, retrieval events, sub-queries, the claim ledger with a v1→v2 diff, citation click-through, live TTFT/retrieval/token counters, and the measured B1 overlay and metrics table. Because the UI displays only what arrives on the telemetry stream, nothing can be shown that is not also logged.

## Starting State

- Phases 2–8 complete: telemetry events with the fields listed in the roadmap, `answer_chunk`/`answer_delta` events, ledger, reconcile telemetry.
- Phase 6: `summary.json`, `report.md`, and `--trace-out` traces (`out/traces/b1_<scenario>.jsonl`).
- FastAPI shell with `/ws/telemetry`, `/metrics`, `/ws/stream`.

## Implementation Scope

**MUST HAVE**
- Transcript panel with per-chunk decision chips (WAIT / PROVISIONAL / COMMIT / SUPPRESS, tier, reason).
- Controller lane and retrieval-event list (query, trigger, `source: fresh|cache`, latency).
- Sub-query panel with per-sub-query evidence and sufficiency scores.
- Answer panel: streamed claims with `[Doc §n]` citations, uncertainty lines.
- Ledger panel with version selector and diff view (added / revised / retracted / preserved) and `text_hash` column.
- Citation click-through: opens the cited chunk text.
- Live counters: TTFT, retrieval calls, LLM calls, tokens, estimated cost, wasted tokens.
- Scenario runner control: choose a scenario, play at 1×, or type/stream free text.
- Endpoints: `/api/scenarios`, `/api/run`, `/api/chunk/{chunk_id}`, `/api/summary`, `/api/trace/{mode}/{scenario}`.
- UI field manifest and contract test.

**HIGH VALUE**
- B1 overlay: the recorded B1 trace for the same scenario on the same timeline.
- Metrics tab reading `summary.json` (Ours vs B0/B1, G1–G6, ablations A1–A5).

**OPTIONAL**
- Inline SVG timeline polish, dark mode, keyboard shortcuts.

## Files / Modules

Create/modify:
- `src/slrag/server/static/index.html`, `app.js`, `style.css`, `fields.json`
- `src/slrag/server/routes.py` (the `/api/*` endpoints)
- `src/slrag/replay/runner.py` (`--trace-out` if not already complete)
- `eval/demo/*.jsonl` (demo scenarios, kept outside the image; mounted read-only in compose)
- `tests/test_ui_contract.py`, `tests/test_api_routes.py`

## Interfaces & Contracts

```
GET  /api/scenarios                → [{id, category, turns, path}]
POST /api/run   {scenario_id, speed} → {session_id}     # plays the scenario through the gateway at 1×
GET  /api/chunk/{chunk_id}          → {chunk_id, cite, doc_id, section, text}
GET  /api/summary                   → contents of out/summary.json (404 if absent)
GET  /api/trace/{mode}/{scenario_id} → JSONL of the recorded baseline trace
WS   /ws/telemetry                  → TelemetryEvent JSON per message (from Phase 2)
```

`fields.json` lists, per event type, every field the UI reads:

```json
{"controller_decision":["tier","decision","query","payload.reason","payload.drift_cos","ts_stream_s"],
 "retrieval_started":["query","trigger","payload.source","payload.sub_query_id","ts_stream_s"],
 "answer_version_transition":["payload.from","payload.to","payload.kept","payload.revised","payload.added","payload.retracted","payload.unchanged_hashes_ok"], "...":[]}
```

## Implementation Steps

1. **Static shell.** One `index.html` with vanilla JS and CSS, no build step, no npm, no CDN dependencies (the page must work offline).
2. **Event store in JS.** One array of received `TelemetryEvent`s per session. All panels are pure render functions over this array, and they re-render incrementally on each message. The UI never computes a metric that is not already in an event, except for display differences such as elapsed time.
3. **Transcript + controller lane.** Map `chunk_received` and the following `controller_decision` to a row: chunk text, decision chip with tier badge, reason on hover, `drift_cos` if present. Render the `utterance_end` marker.
4. **Retrieval panel.** List `retrieval_started`/`retrieval_completed` pairs with trigger, query, latency, `source` badge (fresh / cache), sub-query id. Draw an `utterance_end` line so early retrieval is visually obvious.
5. **Plan and sufficiency panel.** From `plan_completed` show the sub-queries with `source` (`llm`/`fallback`/`single`). From `sufficiency_check` show `dense_top1`, `coverage`, and answerable / insufficient with the thresholds shown next to the values.
6. **Answer panel.** Append `answer_chunk` text as it arrives with cite links. Render `uncertainty_flag` as a distinct amber line with the reason.
7. **Ledger and diff.** Maintain the ledger from `answer_delta` ops. A version dropdown shows v1 / v2. The diff view colors claims: green = added, amber = revised (with the previous text struck through), red = retracted, grey = preserved. A hash column shows the first 6 chars of `text_hash` so identical hashes for preserved claims are visible.
8. **Citation click-through.** Clicking a cite calls `/api/chunk/{chunk_id}` and shows the chunk text in a side drawer, highlighting the sentence with the highest lexical overlap with the claim (client-side, purely presentational).
9. **Counters.** Live: elapsed since `utterance_end`, TTFT (`turn_summary.ttft_ms`), retrieval calls, LLM calls, tokens, estimated cost, wasted tokens (from `turn_summary`). For refinement turns show "Δ retrieval calls / tokens vs restart" using the baseline trace when loaded.
10. **Scenario control.** Dropdown from `/api/scenarios`, Start/Pause/Reset, plus a free-text box that sends chunks through `/ws/stream`. Speed fixed at 1× in the demo build.
11. **B1 overlay (HIGH VALUE).** Load `/api/trace/b1/<scenario>` and draw its events as a second lane on the same timeline (stream time aligned to `utterance_end`). Its TTFT counter is labeled "B1 (recorded, same machine)".
12. **Metrics tab (HIGH VALUE).** Read `/api/summary`. Render Ours vs B0 vs B1 for recall@5, groundedness, TTFT p50/p95, cost per turn, early-retrieval and false-trigger rate, G1–G6 pass/fail, and ablations A1–A5.
13. **Field manifest.** Fill `fields.json` as panels are written. Every field the JS reads must be listed there.
14. **Contract test** (`tests/test_ui_contract.py`): run a scenario through the engine, collect its telemetry, and assert that every field in `fields.json` for each event type is present in at least one emitted event of that type (or explicitly marked optional). Also grep `app.js` for `event.` property accesses and assert each is in `fields.json` (simple static check), which enforces "the UI only shows what is logged".
15. **Compose wiring.** Mount `eval/demo/` into the app container read-only at `/data/demo` in the dev profile only. `/api/scenarios` also lists `/data/replay/*.jsonl`.

## Algorithms / Logic

- **Ledger reconstruction:** `answer_delta.ops` are applied in order to the previous rendered ledger. `keep` leaves a claim as is, `revise` replaces text and cites, `retract` removes from the rendered answer but keeps it in the diff view as red, `add` appends. Versions are stored as snapshots so the dropdown switches instantly.
- **Timeline alignment:** all lanes use `ts_stream_s` relative to the turn start. Baseline overlay events are shifted so their `utterance_end` aligns with Ours.
- **Highlighting:** sentence with max token overlap with the claim, computed client-side, only for display.
- **Failure behavior:** if `/api/trace` or `/api/summary` is missing, the overlay and metrics tab show "not available" and the rest works.

## Integration

- Consumes every telemetry event type produced by Phases 2–8. It changes no engine behavior.
- Reads Phase 6 outputs (`summary.json`, traces) and the Phase 2 WebSocket.
- Phase 10 packages the static files into the image and adds the demo scenarios to the dev profile.

## Verification

```bash
pytest -q tests/test_ui_contract.py tests/test_api_routes.py
docker compose up --build -d
python -m slrag.cli replay eval/demo --mode b1 --trace-out out/traces --speed 1     # record baseline traces once
open http://localhost:8000     # or curl -s localhost:8000/api/scenarios | jq .
```

Acceptance checks (manual checklist, run once per demo scenario):
- Example 1 scenario: the transcript shows WAIT at the first chunk, PROVISIONAL at the second with the query, COMMIT at the third, and three sub-queries appear. The retrieval panel shows `retrieval_started` before the `utterance_end` marker. Claims stream with citations.
- Refinement scenario: the version dropdown shows v1 and v2. The diff shows at least one amber or green claim and grey preserved claims with identical hash prefixes. The counter shows the delta retrieval calls and tokens.
- Suppression scenario: the controller lane shows SUPPRESS with tier T0 and reason `presentation_restructure`. The retrieval panel stays empty for the turn.
- Unanswerable scenario: an amber uncertainty line with gate scores. Clicking a cite opens the chunk text.
- Overlay: with a recorded B1 trace loaded, the two TTFT values are shown side by side and correspond to the numbers in the trace files.
- Metrics tab matches `out/summary.json` values exactly.
- The page works with no internet access (no external requests in the browser network log).
- Automated: `test_ui_contract.py` passes, and `/api/chunk/{id}` returns the stored chunk text for every cite emitted in a run.

## Demo Capability

The full jury-facing demo surface: controller decisions per chunk, early retrieval relative to `utterance_end`, decomposed sub-queries, evidence and sufficiency scores, streaming cited answer, uncertainty, v1→v2 diff with preserved hashes, counters, the B1 overlay, and the metrics table with G1–G6.

## Definition of Done

- [ ] All MUST HAVE panels render from live telemetry only.
- [ ] `fields.json` and the contract test pass. No displayed value is computed outside telemetry.
- [ ] Citation click-through returns the exact stored chunk.
- [ ] Ledger diff correctly shows added / revised / retracted / preserved with hashes.
- [ ] B1 overlay and metrics tab work, or are listed as cut in the roadmap cut order.
- [ ] Works offline in the container. No external assets.
- [ ] Phase 1–8 tests still pass. The engine is untouched.

## Failure / Rollback

- Missing data in a panel: check `fields.json` against the emitted event payloads first, and fix the emitter, not the UI.
- Desync between lanes: check that the baseline trace was recorded from the same scenario and that `utterance_end` alignment is applied.
- WebSocket drops: the JSONL remains the source of truth. Add a `/api/replay-telemetry` read of `out/telemetry.jsonl` only if needed.
- The UI is optional for correctness. If time is short, keep transcript, controller lane, retrieval list, ledger diff and citation click-through, and cut the rest per the roadmap.

## Output

- Static trace UI (HTML/JS/CSS), `fields.json`
- `/api/*` endpoints
- Demo scenarios in `eval/demo/`, recorded B1 traces
- UI contract and API tests
