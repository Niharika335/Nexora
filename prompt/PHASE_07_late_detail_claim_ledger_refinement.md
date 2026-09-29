# Phase 7 — Late-Detail Refinement via Claim-Ledger Delta

**Priority:** MUST HAVE (contradiction guard: HIGH VALUE)

## Objective

Implement "refine, don't restart". When a later turn adds a constraint to an existing answer, the system identifies the affected sub-intents, retrieves only for the delta, rewrites only the affected claims, keeps every other claim byte-identical, and bumps the answer version with a claim-level diff. This is PDF Example 2, gate G5, and the second core differentiator. Ablation A4 measures the savings over restart.

## Starting State

- Phase 5 complete: multi-sub-intent ledger, planner, parallel retrieval, verifier.
- Phase 6 complete: harness, restart baselines B0/B1, metrics, split.
- Ledger has `text_hash` per claim and `answer_version`.

## Implementation Scope

**MUST HAVE**
- Delta planner (one constrained-JSON LLM call): `relation`, `affected_sub_intents`, `delta_queries`.
- Delta retrieval and sufficiency gate on the new evidence.
- Rewriter: `keep | revise | retract` per affected claim, plus new claims; only affected sub-intents' claims are sent to the LLM.
- Verification of revised and new claims only.
- Ledger versioning, `answer_delta` with ops, `answer_version_transition` event, preserved citations.
- Fallbacks: invalid planner JSON, insufficient delta evidence.
- 15 late-detail scenarios and metrics: claim preservation, contradiction rate, refinement savings (A4), G5.

**HIGH VALUE**
- Contradiction guard between revised/new and preserved claims.

**OPTIONAL**
- None.

## Files / Modules

Create/modify:
- `src/slrag/plan/delta_planner.py`
- `src/slrag/answer/rewriter.py`
- `src/slrag/state/delta.py` (apply decisions, versioning, diff)
- `src/slrag/state/ledger.py` (constraints, history, versioning)
- `src/slrag/engine/turn_engine.py` (refinement path)
- `src/slrag/llm/schemas.py`, `prompts.py`
- `src/slrag/replay/metrics.py` (preservation, contradiction, savings), `runner.py` (restart comparison)
- `eval/scenarios/*/late_*.jsonl`
- `tests/test_delta_planner.py`, `test_rewriter.py`, `test_delta_apply.py`, `test_refinement_e2e.py`

## Interfaces & Contracts

Delta planner input: the utterance, `[{id, text}]` sub-intents, and `[{claim_id, sub_intent_id, first_20_words}]`.

```json
// delta planner output schema
{"type":"object","required":["relation","affected_sub_intents","delta_queries"],
 "properties":{
  "relation":{"enum":["modifies","adds","unrelated"]},
  "affected_sub_intents":{"type":"array","items":{"enum":["<current sub-intent ids>"]}},
  "delta_queries":{"type":"array","maxItems":2,"items":{"type":"string","maxLength":160}}}}
```

Rewriter output schema (cites enum = cites of the affected claims ∪ new delta evidence ∪ evidence already held for the affected sub-intents):

```json
{"type":"object","required":["decisions","new_claims"],
 "properties":{
  "decisions":{"type":"array","items":{"type":"object","required":["claim_id","action"],
     "properties":{"claim_id":{"enum":["<affected claim ids>"]},
                   "action":{"enum":["keep","revise","retract"]},
                   "text":{"type":"string","maxLength":320},
                   "cites":{"type":"array","items":{"enum":["<allowed cites>"]}}}}},
  "new_claims":{"type":"array","maxItems":3,"items":{"type":"object","required":["sub_intent_id","text","cites"],
     "properties":{"sub_intent_id":{"enum":["<affected ids>"]},"text":{"type":"string"},
                   "cites":{"type":"array","minItems":1,"items":{"enum":["<allowed cites>"]}}}}}}}
```

`answer_delta` ops: `keep {claim_id}`, `revise {claim_id, text, cites, prev_text_hash}`, `retract {claim_id}`, `add {claim_id, text, cites}`. `change_type`: `refine` (modifies), `add` (adds/unrelated).

Ledger additions: `constraints[]` (`constraint_id`, `text`, `turn_id`, `affects[]`), claim `history[]` entries `{version, action, text, cites}`, `status: active | revised | retracted`.

Telemetry: `plan_completed` (payload `relation`, `affected_sub_intents`, `delta_queries`, `fallback`), `answer_version_transition` (payload `from`, `to`, `kept`, `revised`, `retracted`, `added`, `unchanged_hashes_ok`, `rewrite_input_claim_ids`), retrieval events with `trigger = "refinement"`.

## Implementation Steps

1. **Route.** In the turn engine, for a non-suppressed turn where the session ledger has `answer_version ≥ 1`, call the delta planner at commit or `utterance_end`. A first turn in a session (no ledger) always uses the Phase 5 flow (`change_type: "initial"`).
2. **Delta planner.** One `json_call`, 1.5 s timeout. Prompt (generic): decide whether the new statement modifies an existing sub-question, adds a new one, or is unrelated; name affected sub-intent IDs; propose ≤ 2 short retrieval queries that capture only what is new.
3. **Validate.** IDs must exist in the ledger. On invalid JSON, timeout, or empty `affected_sub_intents` for `modifies`, apply the fallback: `relation = "adds"`, the utterance itself as the single query (log `delta_fallback`). Never restart the session.
4. **Constraint record.** Store the utterance's constraint text as a `Constraint` with `affects` = the affected sub-intent IDs.
5. **`modifies` path.**
   - Form each delta query as `"<constraint text> <affected sub-query text>"` (or the planner's queries, each with the affected sub-query text appended if missing).
   - Retrieve with `trigger = "refinement"`, using the Phase 4 cache first.
   - Merge the new evidence into the session evidence store.
   - Run the sufficiency gate on the delta evidence for each affected sub-intent.
   - If sufficient: call the rewriter with the affected sub-intents' claims (text and cites), the new evidence and the constraint.
   - If insufficient: keep old claims unchanged and add an uncertainty item `"could not verify how <constraint> affects <sub-intent>"` with reason `refinement_insufficient`.
6. **`adds` / `unrelated` path.** Run the Phase 5 flow (planner gate → sub-queries → retrieval → gate → draft → verify) for the new content only. Append new `SubIntent`s to the ledger. Existing claims are untouched. `change_type = "add"`.
7. **Rewriter.** One `json_call` with the enum-constrained schema. The prompt shows only the affected claims, the constraint and the new evidence (≤ 4 chunks per affected sub-intent). It returns per claim `keep | revise | retract`, plus optional new claims.
8. **Verify** revised and new claims with the Phase 3 verifier. A revised claim that fails verification reverts to its previous text (status `active`, logged `revision_rejected`). A new claim that fails is dropped.
9. **Apply** (`state/delta.py`). For `keep`: no change. For `revise`: update text/cites, `status = "revised"`, `last_modified_version = n+1`, new `text_hash`, append history. For `retract`: `status = "retracted"`. New claims get new IDs and `created_version = n+1`. Claims of unaffected sub-intents are never touched. Increment `answer_version` once.
10. **Preservation assertion.** After applying, recompute `text_hash` for every claim outside the affected set and every `keep`. Assert equality with the pre-refinement hashes. Put the result in `answer_version_transition.payload.unchanged_hashes_ok`. A failed assertion is a bug, so raise in tests and log an error in production.
11. **Emit.** `answer_delta` with ops and `rendered_answer` (active claims in sub-intent order). Stream `answer_chunk` events only for added and revised claims (`first_token` on the first one emitted). Kept claims are not re-streamed.
12. **Contradiction guard (HIGH VALUE).** For each revised or new claim, compare against preserved claims that share ≥ 2 content lemmas. Extract `(noun lemma, number)` pairs via spaCy `nummod`. If the same noun has a different number, flag `contradiction_flag`, downgrade the new/revised claim to an uncertainty item, and keep the preserved claim.
13. **Late-detail scenarios.** Author 15 two-turn scenarios (turn 1 content, turn 2 constraint) with gold: `affected_sub_intents`, `unaffected_claims_expected: true`, and the gold cites for the delta. Add them to `eval/scenarios/`, extend `eval/split.json` (stratified, seed 13, ~50/50).
14. **A4 experiment.** Extend the runner: for each late-detail scenario also run the restart baseline (B1 with restart on late detail) and record retrieval calls and tokens for turn 2. Metrics:
    - `preservation_rate` = unaffected claims byte-identical / unaffected claims.
    - `contradiction_rate` = turns with a post-refinement contradiction (human-labeled on the test sample; the guard's count is reported separately).
    - `refinement_savings` = 1 − (Ours retrieval calls, tokens) / (restart retrieval calls, tokens).
15. **G5 in the report.** G5 passes if `preservation_rate = 1.0`, no session was cleared or fully re-searched on a late-detail turn, and version lineage is complete (every version has a transition event).

## Algorithms / Logic

- **Relation semantics:** `modifies` = the utterance changes the conditions of an existing sub-intent (dates, exceptions, scope). `adds` = a new question in the same session. `unrelated` = a new topic. `adds` and `unrelated` both take the Phase 5 flow, and differ only in telemetry labels.
- **`text_hash`** = `sha256(text + "|" + "|".join(sorted(cites)))`.
- **Version rule:** `answer_version` increases by exactly 1 per refinement or add turn, never for presentation turns.
- **What may reach the rewrite LLM:** only claims whose `sub_intent_id ∈ affected_sub_intents`. This is logged as `rewrite_input_claim_ids` and asserted in tests.
- **Retrieval savings source:** delta retrieval touches only the ≤ 2 delta queries. A restart would re-run every sub-query. Savings are measured, not assumed.
- **Fail-safe order:** on any error in the refinement path, keep the previous ledger version unchanged, emit an uncertainty item, and never partially apply.

## Integration

- Builds on the Phase 3 ledger and verifier, the Phase 5 flow and Phase 4 cache/presenter routing (T0 still runs first).
- Uses the Phase 6 harness for A4, the restart baselines and G5.
- Phase 8 reuses the delta planner and rewriter for within-turn reconcile. Phase 9 renders `answer_delta` ops and hashes.

## Verification

```bash
pytest -q tests/test_delta_planner.py tests/test_rewriter.py tests/test_delta_apply.py
pytest -q -m integration tests/test_refinement_e2e.py
python scripts/validate_scenarios.py eval/scenarios
python -m slrag.cli replay eval/scenarios/test --mode ours --category late_detail --out out/ours_late.jsonl --speed 1
python -m slrag.cli replay eval/scenarios/test --mode b1   --category late_detail --out out/b1_late.jsonl --speed 1
python -m slrag.cli report out/ --split test --experiment A4
```

Acceptance checks:
- **Delta apply unit test:** a ledger with 4 claims in 2 sub-intents, decisions `keep, revise` on sub-intent 1, none on sub-intent 2 → claim hashes for sub-intent 2 and the kept claim are unchanged, the revised claim's hash changes, version 1 → 2.
- **Rewrite-scope test:** `rewrite_input_claim_ids` contains only claims of the affected sub-intents.
- **Example 2 e2e:** turn 1 "Summarize the <policy topic> for <case>" gives v1 with cites. Turn 2 "<the case was international and the booking was made after travel>" gives v2 with the base claims preserved, ≥ 1 revised or added claim with delta citations, and no full-corpus re-search: `retrieval_calls ≤ 2` in turn 2 versus the restart baseline's retrieval count for the same scenario.
- **Enum test:** the rewriter cannot cite a chunk outside its allowed set (schema-level and verifier check (a)).
- **Fallbacks:** invalid planner JSON → `delta_fallback`, `relation = adds`, ledger v1 claims still present. Insufficient delta evidence → v1 claims unchanged plus a `refinement_insufficient` uncertainty item, and no new claim.
- **Contradiction guard test:** fixture with preserved "up to 30 attendees" and a new claim "up to 50 attendees" for the same noun → `contradiction_flag` and the new claim is downgraded.
- **Presentation after refinement:** "Repeat that in two bullets" after v2 uses only active claims and does not change the version.
- **A4 on the test split:** `preservation_rate = 1.0`. Report retrieval calls and tokens for Ours vs restart with the ratio.
- **Coverage:** `trace_coverage = 1.0` including `answer_version_transition` for every refinement turn.

## Demo Capability

The refinement scene: a base answer v1, then a spoken constraint. The delta plan JSON is visible, only the affected claims change, the others stay identical (same hashes), v1→v2 lineage is logged, and a counter compares delta retrieval calls and tokens with a measured restart.

## Definition of Done

- [ ] Delta planner and rewriter enum-constrained and tested, with fallbacks.
- [ ] Only affected sub-intents' claims reach the rewrite LLM (test asserts it).
- [ ] Preservation assertion passes on all late-detail scenarios (`preservation_rate = 1.0`).
- [ ] Version lineage complete: every version has an `answer_version_transition`.
- [ ] Insufficient delta evidence and invalid JSON do not clear or corrupt the ledger.
- [ ] 15 late-detail scenarios added and split. A4 and G5 in the report.
- [ ] Contradiction guard implemented, or explicitly deferred and listed in the report (HIGH VALUE item).
- [ ] Phase 1–6 tests still pass. Non-late-detail behavior unchanged.

## Failure / Rollback

- Rewriter changes claims it should keep: tighten the prompt and check the scope test. The guarantee for unaffected sub-intents is structural (they are never sent), so a failure there is a code bug.
- Over-broad `affected_sub_intents`: inspect `plan_completed` and the sub-intent texts sent to the planner. Shorten the claim one-liners if needed.
- Pronoun-only late details ("that too"): expect low relation accuracy. Record it as an edge-case failure for the report rather than special-casing it.
- If refinement is unstable, set `refinement.enabled: false` so the engine treats late details as `adds` on the same session, still preserving state and never restarting.

## Output

- Delta planner, rewriter, ledger delta engine with versioning and diff
- Contradiction guard
- 15 late-detail scenarios, updated split
- A4 results, G5 status, preservation/contradiction/savings metrics
