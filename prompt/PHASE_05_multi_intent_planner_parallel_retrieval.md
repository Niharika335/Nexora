# Phase 5 — Multi-Intent Planner, Parallel Retrieval & Per-Sub-Intent Uncertainty

**Priority:** MUST HAVE

## Objective

Handle compound utterances. Decompose them into search-ready, self-contained sub-queries (≤ 4), retrieve in parallel with quotas, run the sufficiency gate and drafting per sub-intent, and stream one unified answer with per-sub-intent citations and explicit uncertainty. This delivers PDF Example 1 fully and gate G3, and guards against the "over-fragmentation" pitfall.

## Starting State

- Phase 4 complete: streaming controller with `commit_hint.multi_intent`, evidence cache, streaming turn state machine, Presenter.
- Phase 3 gate, drafter, verifier, ledger and emitter available.
- Phase 1 `apply_quota` implemented and tested.

## Implementation Scope

**MUST HAVE**
- Multi-intent gate (deterministic, liberal).
- Query Planner decompose mode: one constrained-JSON LLM call, 1.5 s timeout.
- spaCy coordination splitter fallback with slot injection.
- Post-processing: slot injection check, merge (cos ≥ 0.92), min-token drop, cap at 4.
- Parallel retrieval per sub-query, cache-aware, with per-sub-query quotas.
- Per-sub-intent gate, drafting (parallel), verification, uncertainty.
- Emission in sub-intent order as each sub-intent finishes.
- Final `TurnOutput` with `sub_queries` and the multi-sub-intent ledger.

**HIGH VALUE**
- Planner `plan_completed` telemetry with per-sub-query cache reuse flags.

**OPTIONAL**
- Priority ordering of sub-intents by utterance order and importance (utterance order is enough).

## Files / Modules

Create/modify:
- `src/slrag/plan/planner.py`, `splitter.py`, `multi_intent_gate.py`
- `src/slrag/llm/schemas.py` (planner schema), `prompts.py` (planner prompt)
- `src/slrag/engine/turn_engine.py` (final and commit paths use the planner)
- `src/slrag/answer/emitter.py` (multi-sub-intent ordering)
- `src/slrag/state/ledger.py` (multiple sub-intents, uncertainty items)
- `tests/test_multi_intent_gate.py`, `test_planner.py`, `test_splitter.py`, `test_parallel_retrieval.py`, `test_e2e_multi_intent.py`

## Interfaces & Contracts

```python
class SubQuery(BaseModel): id:str; text:str                          # "q1".."q4"
class Plan(BaseModel):     sub_queries:list[SubQuery]; source:Literal["llm","fallback","single"]; shared_slots:dict; dropped:list[dict]; merged:list[dict]

def multi_intent_gate(buffer:str) -> GateDecision                     # {run_planner:bool, reasons:list[str]}
async def plan(buffer:str, slots:dict, *, timeout_s:float) -> tuple[Plan, Usage | None]
def split_fallback(buffer:str, slots:dict) -> list[str]

async def retrieve_all(plan:Plan, *, trigger:str, cache:EvidenceCache) -> dict[str, list[Evidence]]   # keyed by sub_query id, after quota
```

Planner LLM JSON Schema:

```json
{"type":"object","required":["sub_queries"],
 "properties":{"sub_queries":{"type":"array","minItems":1,"maxItems":4,
   "items":{"type":"string","maxLength":160}}}}
```

Telemetry: `plan_completed` payload `{source, sub_queries:[{id,text,cache:"hit"|"miss"}], merged, dropped, latency_ms, tokens_in, tokens_out}`. Retrieval events use `trigger = "multi_intent"` at commit and `"final"` otherwise, with `sub_query_id`.

## Implementation Steps

1. **Multi-intent gate** (§Algorithms). Implement with spaCy noun-chunks, coordination detection and request-clause detection. It is liberal on purpose: false positives cost one LLM call, false negatives cost G3.
2. **Shared slots.** From Phase 4 slot extraction, compute `shared_slots` (place, quantity, date, organization) present in the buffer. These are the context that each sub-query must carry.
3. **Planner prompt.** Generic instructions: split the request into independent, self-contained search questions; each must contain the shared context (place, quantity, date); do not split a single question; maximum four; no answers. Few-shot exemplars use an unrelated synthetic domain.
4. **Planner call.** Use `json_call` with `asyncio.wait_for(timeout=1.5)`. On timeout, exception or invalid JSON call `split_fallback` and set `source: "fallback"`, emit a `plan_fallback` flag in the payload.
5. **Post-processing** (§Algorithms): slot injection check, min-token drop, cosine merge, cap.
6. **Single-intent path.** If the gate says no planner, the plan is `[q1 = Q_final]`, `source: "single"`, and no LLM call is made.
7. **Parallel retrieval.** `retrieve_all` runs `instrumented_search` per sub-query with `asyncio.gather`. For each, first ask the cache (Phase 4 rule). Then apply `apply_quota` across the sub-queries (4 each, cap 12). Emit one `retrieval_started` and `retrieval_completed` per sub-query, all sharing the dispatch timestamp.
8. **Per-sub-intent processing.** For each sub-query concurrently: gate → draft (skipped if insufficient) → verify. Draft calls are limited by the LLM semaphore (3 concurrent, additional requests queue).
9. **Ordered emission.** Await results in sub-intent order. Emit a sub-intent's `answer_chunk`s as soon as it and all previous ones are done, which lowers TTFT compared with waiting for all.
10. **Ledger.** Create one `SubIntent` per sub-query with its gate scores. Uncertainty items are `{sub_intent_id, reason: "insufficient_evidence" | "no_verified_claims", gate_scores}`.
11. **Final record.** `TurnOutput.sub_queries` = plan texts. `citations` = deduplicated cites of all emitted claims in order. `uncertainty` = joined per-item sentences.
12. **Wire the controller.** At COMMIT with `multi_intent` (and at final if no COMMIT happened) call `plan`. At COMMIT the planner result is stored on the turn (`committed_plan`) and retrieval is dispatched per sub-query. At `utterance_end`, if the final buffer's tail adds no new anchors, reuse the committed plan; otherwise re-plan from the final buffer (Phase 8 later replaces this with reconcile).
13. **Tests.** Add compound fixtures (2, 3, 4 intents), single-intent long utterances, and near-duplicate phrasings.

## Algorithms / Logic

**Multi-intent gate** (`run_planner` is true if any of):
- ≥ 2 coordinated topic noun-phrases exist (noun-chunks linked by `and`, `,`, `as well as`, `plus`),
- ≥ 2 request clauses (verbs like `need, want, know, tell, check, find, explain, list`, with distinct objects),
- content tokens ≥ 14 with a list cue (`and`, `also`, `plus`, comma-separated items).
Otherwise `run_planner = false`.

**Post-processing:**
1. **Slot injection check.** For each sub-query, each shared slot value missing from its text is appended (`"<sub-query> <slot value>"`) so every sub-query is self-contained.
2. **Drop** sub-queries with < 3 content tokens (`dropped`, reason `too_short`).
3. **Merge** sub-query pairs with `cos ≥ 0.92` (keep the earlier, record `merged`). This is the over-fragmentation guard.
4. **Cap** at 4, keeping utterance order.
5. If everything was dropped or merged away, fall back to the single-query plan `[q1 = Q_final]`.

**Fallback splitter (`split_fallback`):** parse the buffer with spaCy. Find the coordinated noun-chunks under the last request verb (`conj` chain), and split clauses at `and`/`,` where both sides contain a noun-chunk. Each part becomes `"<part> <shared slots>"`. If no coordination is found, return `[Q_final]`.

**Retrieval and quota:** per sub-query BM25 top-30 + dense top-30 → RRF → dedupe → top-10 → quota 4, global 12 (round-robin by rank when capped). Chunks may appear in more than one sub-query's list. Only dedupe within a sub-query, so each sub-intent keeps the evidence it needs. The ledger's evidence store is the union.

**Per-sub-intent gate:** as in Phase 3, each with its own `dense_top1` and `coverage`, computed on its own sub-query.

## Integration

- Extends the Phase 4 turn engine: COMMIT (multi-intent) and final path use the planner.
- Reuses the Phase 4 cache for sub-query reuse, and the Phase 3 gate, drafter, verifier and emitter per sub-intent.
- Phase 6 measures sub-intent recall/over-fragmentation. Phase 7 attaches constraints to sub-intent IDs created here. Phase 8 moves drafting to commit time using this plan.

## Verification

```bash
pytest -q tests/test_multi_intent_gate.py tests/test_planner.py tests/test_splitter.py tests/test_parallel_retrieval.py
pytest -q -m integration tests/test_e2e_multi_intent.py
python -m slrag.cli simulate tests/fixtures/streams/compound_3intent.jsonl --url ws://localhost:8000/ws/stream
python -m slrag.cli trace out/telemetry.jsonl --turn t1 --show plan,retrieval,uncertainty
```

Acceptance checks:
- **Gate table test:** 10 compound and 10 single-intent fixture utterances → correct `run_planner` on ≥ 18/20. A long single-intent utterance (≥ 14 tokens, no coordination) → `run_planner = false` and `llm_calls == 0` in planning.
- **Planner output test (integration):** 3-intent fixture returns 3 sub-queries, each containing the shared slots (place, quantity).
- **Over-fragmentation guard:** a single question phrased in two near-identical ways yields 1 sub-query after merge.
- **Timeout fallback:** with `planner.timeout_s: 0.001` in a test config, plan source is `fallback`, `plan_fallback` is logged, and a 2-intent fixture still yields ≥ 2 sub-queries.
- **Parallelism:** in a 3-sub-query turn, the three `retrieval_started` timestamps are within 20 ms of each other.
- **Quota:** ≤ 4 evidence chunks per sub-query and ≤ 12 total. A dominant sub-query does not remove the other sub-queries' evidence.
- **Uncertainty:** an utterance whose third topic is absent from the corpus gives 2 answered sub-intents, 1 `uncertainty_flag`, and the final `uncertainty` string names the missing sub-intent. No claim is emitted for it.
- **Ordering:** `answer_chunk` events follow sub-intent order. `first_token` is on the first emitted claim.
- **PDF-shape check:** `TurnOutput` keys equal the PDF record (`retrieval_events`, `sub_queries`, `answer`, `citations`, `uncertainty`) plus `meta`.
- **Coverage:** trace coverage has no missing events for multi-intent turns.

## Demo Capability

Example 1 end to end: an utterance with three needs streams in, early retrieval fires, three sub-queries appear, parallel retrieval bars run, and a single answer with per-sub-intent citations streams out, with an explicit uncertainty line for the sub-intent the corpus cannot support.

## Definition of Done

- [ ] Multi-intent gate table passes and single-intent utterances make no planner call.
- [ ] Planner (LLM) and fallback (timeout) both produce valid, self-contained sub-queries.
- [ ] Merge, drop, cap and slot-injection rules tested.
- [ ] Parallel retrieval with quotas works. Sub-queries are cache-aware.
- [ ] Per-sub-intent uncertainty verified. No claims emitted for insufficient sub-intents.
- [ ] Output record matches the PDF shape.
- [ ] Phase 1–4 tests still pass. Single-intent behavior is unchanged from Phase 4.

## Failure / Rollback

- Planner returns unusable JSON: the fallback path must take over. Check `plan_fallback` frequency.
- Over-fragmentation: inspect merged and dropped lists in `plan_completed`. Adjust the gate before touching the planner prompt.
- Sub-queries lose context ("catering options" with no place): check the slot injection step.
- Latency spike: check whether the LLM semaphore is saturated and whether the planner call is on the critical path only for multi-intent turns.
- If the planner is unstable, set the gate to `run_planner = false` in config. The system reverts to single-query streaming behavior from Phase 4.

## Output

- Multi-intent gate, planner (LLM + spaCy fallback), post-processing
- Parallel per-sub-query retrieval with quotas
- Per-sub-intent draft/verify/uncertainty and ordered emission
- Multi-sub-intent ledger, PDF-shaped output record
- Compound fixtures and tests
