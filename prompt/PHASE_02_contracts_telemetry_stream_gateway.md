# Phase 2 — Event Contracts, Telemetry Bus & Stream Gateway

**Priority:** MUST HAVE

## Objective

Define every data contract once, and build the telemetry bus and the stream gateway. Telemetry is a scored gate (G6), the UI and harness both read it, and every later component emits into it. If contracts and telemetry are settled now, the later phases add behavior, not plumbing.

## Starting State

- Phase 1 complete: retriever, config, chunk IDs, CLI.
- `pytest -q` green.

## Implementation Scope

**MUST HAVE**
- Pydantic v2 contracts for all frozen data contracts.
- Telemetry bus: async queue, JSONL writer, in-memory aggregate, `/metrics`.
- Stream Gateway: alias normalization, cumulative/delta merge, `seq` ordering, wall-time stamping.
- Stream simulator (real-time paced JSONL playback).
- FastAPI shell: `/health`, `/ws/stream`, `/ws/telemetry`, `/metrics`, `POST /debug/search`.
- Trace coverage checker with the required event table.
- Retrieval telemetry around the Phase 1 retriever.

**HIGH VALUE**
- Cost helper using the reference rates.

**OPTIONAL**
- `POST /session/{id}/chunk` HTTP alternative to WebSocket.

## Files / Modules

Create:
- `src/slrag/schemas.py` (extend)
- `src/slrag/telemetry/bus.py`, `events.py`, `metrics.py`, `coverage.py`
- `src/slrag/stream/gateway.py`, `simulator.py`
- `src/slrag/server/app.py`, `routes.py`
- `src/slrag/engine/turn_engine.py` (ingest-only skeleton: buffers chunks, logs, emits `turn_summary`)
- `src/slrag/state/session.py` (session registry with `session_id`, `asyncio.Lock`, idle TTL)
- `docs/telemetry_schema.md` (generated from the pydantic models by `slrag schema`)
- `tests/test_gateway.py`, `test_telemetry.py`, `test_coverage.py`, `test_server_stream.py`, `tests/fixtures/streams/*.jsonl`

## Interfaces & Contracts

All models in `schemas.py`. Field names are exactly as frozen.

```python
class TranscriptChunk(BaseModel):  type: Literal["transcript_chunk"]; session_id; turn_id; seq:int; ts_s:float; text:str; is_final:bool=False
class UtteranceEnd(BaseModel):     type: Literal["utterance_end"]; session_id; turn_id; seq:int; ts_s:float
class SessionEnd(BaseModel):       type: Literal["session_end"]; session_id

class RetrievalEvent(BaseModel):   retrieval_id; timestamp_s; query; trigger: Literal["provisional","multi_intent","final","refinement"]; sub_query_id: str|None; source: Literal["fresh","cache"]
class Evidence(BaseModel):         # from Phase 1
class Claim(BaseModel):            claim_id; sub_intent_id; text; cites:list[str]; evidence_chunk_ids:list[str]; status: Literal["active","revised","retracted"]
                                   support:dict; created_version:int; last_modified_version:int; text_hash:str; history:list[dict]
class SubIntent(BaseModel):        id; text; status: Literal["answerable","insufficient","pending"]; sufficiency:dict
class Constraint(BaseModel):       constraint_id; text; turn_id; affects:list[str]
class Ledger(BaseModel):           session_id; answer_version:int; sub_intents; constraints; claims; evidence_ids; uncertainty:list[dict]
class AnswerChunk(BaseModel):      type: Literal["answer_chunk"]; turn_id; answer_version; claim_id; text; cites; first_token:bool; ts_s:float
class AnswerDelta(BaseModel):      type: Literal["answer_delta"]; turn_id; answer_version; prev_version:int|None; change_type: Literal["initial","refine","add","presentation"]; ops:list[dict]; rendered_answer:str; uncertainty:list[dict]
class TurnOutput(BaseModel):       retrieval_events; sub_queries; answer; citations; uncertainty:str|None; meta:dict
class TelemetryEvent(BaseModel):   event_id; ts_wall; ts_stream_s; session_id; turn_id; event_type; component; tier:str|None; decision:str|None; latency_ms:float|None; query:str|None; trigger:str|None; evidence_ids:list[str]; answer_version:int|None; cache_hit:bool|None; tokens_in:int; tokens_out:int; est_cost_usd:float; model_id:str|None; cfg_hash:str; payload:dict

class TelemetryBus:
    def emit(self, event_type: str, *, session_id, turn_id, component, **fields) -> TelemetryEvent   # non-blocking
    def subscribe(self) -> AsyncIterator[TelemetryEvent]
    def aggregate(self) -> dict            # counts by event_type, latency p50/p95 per component, tokens, cost
class StreamGateway:
    def ingest(self, raw: dict) -> list[TranscriptChunk | UtteranceEnd | SessionEnd]   # ordered, normalized
```

WebSocket protocol: client → `/ws/stream` sends raw JSON events. Server → `/ws/telemetry` pushes `TelemetryEvent` JSON, one per message.

## Implementation Steps

1. **Contracts.** Implement all models with strict types. Add `slrag schema` to dump JSON Schema to `docs/telemetry_schema.md`.
2. **Telemetry bus.**
   - `emit` builds the event (uuid `event_id`, UTC `ts_wall`, `cfg_hash` from config) and `put_nowait` onto an `asyncio.Queue(maxsize=100_000)`.
   - A writer task drains the queue, appends `orjson` lines to `${OUT_DIR:-out}/telemetry.jsonl`, flushes every 50 events or 200 ms, and fans out to WS subscribers.
   - On overflow, drop nothing silently: log a `telemetry_overflow` counter in the aggregate and stderr.
3. **Cost helper.** `est_cost_usd = (tokens_in·ref_in + tokens_out·ref_out)/1e6` from config.
4. **Aggregate.** Maintain counters by `event_type`, per-component latency reservoir (p50/p95), total tokens and cost. Serve at `GET /metrics`.
5. **Gateway.** Normalize keys and merge chunks per §Algorithms. Convert `is_final` or a `type: "end"|"utterance_end"` message into `UtteranceEnd`.
6. **Session registry.** `session.py` creates sessions lazily, holds `asyncio.Lock`, tracks `last_seen`, and a sweeper deletes sessions idle > `session.idle_ttl_s`. `session_end` deletes immediately. No cross-session state exists.
7. **Turn engine skeleton.** For each normalized chunk emit `chunk_received` (payload: buffer text, merge mode). On `utterance_end` emit `turn_summary` with `retrieval_calls=0`, `llm_calls=0`, `ttft_ms=null`. This skeleton is replaced in Phase 3 and 4, not thrown away.
8. **Retrieval telemetry.** Add `instrumented_search(query, trigger, ...)` around the Phase 1 retriever. It emits `retrieval_started` and `retrieval_completed` (latency, evidence IDs) and returns evidence with a `retrieval_id`.
9. **Server.** FastAPI app with the endpoints above. `POST /debug/search {query}` runs `instrumented_search` for demo and testing.
10. **Simulator.** `slrag simulate <file.jsonl> --url ws://... --speed 1.0` reads scenario events with `ts_s`, sleeps to reproduce real timing, and sends them. `--speed` other than 1 is for dev only.
11. **Coverage checker.** `coverage.py` holds the table of required events per turn kind (below) and `check(jsonl) -> {turn_id: missing_types}`. Add `slrag coverage out/telemetry.jsonl`.

## Algorithms / Logic

**Key aliasing (gateway):** `ts_s` ← `ts_s | timestamp_s | timestamp | t`; `text` ← `text | chunk | transcript | content`; `turn_id` ← `turn_id | turn | utterance_id` (default: `t1` for the first turn per session, incremented after each `utterance_end`); `session_id` ← `session_id | session | conversation_id` (default `default`). Unknown extra keys are preserved in a `raw` field, not dropped.

**Cumulative vs delta:** let `norm(x)` be lowercase, whitespace collapsed, leading/trailing `…`/`...` stripped. If `norm(new)` starts with `norm(buffer)` and is longer, replace the buffer (cumulative). Otherwise append with one space (delta). Log `merge_mode` in the payload.

**Ordering:** keep `next_seq` per turn. Hold out-of-order chunks in a reorder buffer for up to 200 ms; then release in `seq` order and emit a `seq_gap` note in the payload. Duplicate `seq` is dropped and counted.

**Required events per turn kind (initial table, extended in later phases):**
- Content turn: `chunk_received`, `retrieval_started`, `retrieval_completed`, `turn_summary` (later: `controller_decision`, `sufficiency_check`, `draft_completed`, `verification`, `answer_emitted`).
- Suppressed turn (Phase 4): `chunk_received`, `controller_decision`, `suppression`, `answer_emitted`, `turn_summary`.
- Refinement turn (Phase 7): plus `plan_completed`, `answer_version_transition`.
- Any turn with an insufficient sub-intent: plus `uncertainty_flag`.

`turn_summary` payload fields (used by the harness): `ttft_ms`, `retrieval_calls`, `llm_calls`, `embed_calls`, `tokens_in`, `tokens_out`, `est_cost_usd`, `wasted_tokens`.

## Integration

- Wraps the Phase 1 retriever with telemetry.
- Provides `TelemetryBus`, contracts and the session registry that Phase 3 onwards import.
- The turn engine skeleton is the file Phase 3 and 4 extend.

## Verification

```bash
pytest -q tests/test_gateway.py tests/test_telemetry.py tests/test_coverage.py tests/test_server_stream.py
uvicorn slrag.server.app:app --port 8000 &
python -m slrag.cli simulate tests/fixtures/streams/two_chunk_turn.jsonl --url ws://localhost:8000/ws/stream
curl -s localhost:8000/metrics | jq .
curl -s -XPOST localhost:8000/debug/search -d '{"query":"<corpus question>"}' -H 'content-type: application/json'
python -m slrag.cli coverage out/telemetry.jsonl
```

Acceptance checks:
- Gateway table test: at least 8 raw payload variants (different key names, cumulative vs delta text, `is_final` end marker) normalize to the expected `TranscriptChunk`/`UtteranceEnd`.
- Cumulative test: chunks `"I need to plan"`, `"I need to plan a workshop in"` → buffer is the second text. Delta test: `"…Pune for 30"`, `"and I need…"` → appended.
- Reorder test: seq 1, 3, 2 arrive → released 1, 2, 3. A missing seq 2 is released after 200 ms with a `seq_gap` note. Duplicate seq 1 is dropped.
- Telemetry burst: emit 10,000 events in a tight loop → JSONL has exactly 10,000 lines, `emit` p95 < 0.2 ms.
- WS parity: the set of `event_id` received on `/ws/telemetry` equals the set in the JSONL for the same run.
- Simulator timing: a scenario with chunks at 0.0 / 0.8 / 1.6 / 2.1 s produces `chunk_received` wall times within ±50 ms of those offsets at `--speed 1`.
- `/metrics` returns counts by event type and latency percentiles.
- The coverage checker reports a deliberately truncated fixture trace as missing events, and a full one as clean.

## Demo Capability

Stream a timestamped transcript in real time and watch normalized chunks, retrieval events and per-turn summaries appear live in the JSONL and over the WebSocket. This is the ingestion and observability half of the pipeline.

## Definition of Done

- [ ] All contracts exist as pydantic models, with `docs/telemetry_schema.md` generated from them.
- [ ] Gateway, ordering and merge tests pass.
- [ ] Telemetry has zero loss under the burst test. JSONL and WS agree.
- [ ] `/metrics`, `/health`, `/debug/search` work.
- [ ] Coverage checker exists and is tested.
- [ ] Session registry expires idle sessions and isolates state. A test proves two sessions never share buffers.
- [ ] Phase 1 tests still pass.

## Failure / Rollback

- Dropped events: check queue size and the writer flush loop. Never fall back to `print`.
- Chunk merge mistakes: log both raw and merged text in the payload, and adjust normalization rather than special-casing test streams.
- If the WebSocket layer misbehaves, the JSONL path remains the source of truth. The UI later reads the same events.
- Phase 1 CLI and index must remain unchanged.

## Output

- `schemas.py` with all frozen contracts
- Telemetry bus, aggregate, `/metrics`, coverage checker
- Stream gateway, simulator, session registry
- FastAPI shell and turn engine skeleton
- Generated telemetry schema doc, fixtures, tests
