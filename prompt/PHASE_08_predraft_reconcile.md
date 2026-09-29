# Phase 8 — Commit-Stage Pre-Drafting & Reconcile

**Priority:** HIGH VALUE (the main source of the measurable TTFT gain; a cut candidate only if time runs out, see the roadmap cut order)

## Objective

Move planning, retrieval and drafting off the critical path. At COMMIT, while the user is still speaking, run the planner, retrieval, gate and drafting, and hold the drafts. At `utterance_end`, reconcile the final text against the committed one. If nothing material was added, verify and stream immediately. If the tail added content, refine the held drafts with the delta machinery. Waste from wrong speculation is counted in cost. This is the first core differentiator and ablation A3.

## Starting State

- Phase 5: planner, parallel retrieval, per-sub-intent gate/draft/verify.
- Phase 7: delta planner, rewriter, `state/delta.py`.
- Phase 6: harness, B1 baseline, TTFT and cost metrics.
- Phase 4: COMMIT stage currently dispatches only retrieval, and drafting waits for `utterance_end`.

## Implementation Scope

**MUST HAVE**
- Pending-draft state per turn (held claims, not yet in the ledger).
- Commit handler: plan → retrieve → gate → draft, in the background.
- Reconcile at `utterance_end`: commit stands / within-turn refinement / full fallback.
- Draft validity rule using the cache reuse rule.
- Waste accounting: wasted tokens, wasted-speculation rate, discarded pre-drafts.
- `predraft_late` handling: wait for in-flight drafts, never emit incomplete claims.
- Experiment A3 with three arms.

**HIGH VALUE**
- Telemetry payloads for the UI: `predraft_ready_ts`, `reconcile_outcome`.

**OPTIONAL**
- Second COMMIT after reconcile (not allowed by the per-turn cap of 1, so out of scope).

## Files / Modules

Create/modify:
- `src/slrag/engine/predraft.py` (pending state and background pipeline)
- `src/slrag/engine/reconcile.py`
- `src/slrag/engine/turn_engine.py` (commit and end-of-utterance paths)
- `src/slrag/state/ledger.py` (allow draft-state claims to become v1 on emission)
- `src/slrag/replay/metrics.py` (waste, TTFT-by-arm)
- `config.yaml` (`speculation` block)
- `tests/test_predraft.py`, `test_reconcile.py`, `test_speculation_waste.py`, `test_ttft_arms.py`

## Interfaces & Contracts

```python
class PendingDraft(BaseModel):
    plan: Plan
    sub_intents: list[SubIntent]                 # gate results included
    evidence: dict[str, list[Evidence]]          # per sub-query
    claims: dict[str, list[DraftClaim]]          # per sub-intent, unverified
    committed_text: str; committed_emb; committed_slots: dict
    ready: dict[str, asyncio.Future]             # per sub-intent draft completion
    tokens_in: int; tokens_out: int; epoch: int

class ReconcileOutcome(str, Enum): STANDS="stands"; REFINED="refined"; REDONE="redone"

async def start_predraft(turn: TurnState, session) -> PendingDraft
async def reconcile(turn: TurnState, pending: PendingDraft | None, final_buffer: str, session) -> ReconcileOutcome
```

Config:

```yaml
speculation: {provisional: true, predraft: true}
```

Telemetry (fields added, no new event types): `draft_completed.payload = {speculative: true, sub_intent_id, ready_ts_s}`; `turn_summary.payload` adds `reconcile_outcome`, `predraft_ready_before_end: bool`, `wasted_tokens`, `wasted_provisional`, `discarded_predrafts`.

## Implementation Steps

1. **Pending state.** On COMMIT, the turn engine creates a `PendingDraft` holding the plan and a per-sub-intent `Future` for each draft. Nothing is written to the ledger and nothing is emitted.
2. **Background pipeline** (`start_predraft`). Run as an `asyncio.Task` under the turn's epoch:
   - plan (Phase 5, gate → LLM planner or single),
   - `retrieve_all` (cache-aware, quotas),
   - per-sub-intent gate,
   - per-sub-intent drafting in parallel (LLM semaphore of 3).
   Each `draft_completed` event carries `speculative: true`. Token usage is added to `PendingDraft.tokens_in/out`.
3. **Epoch safety.** If the turn's epoch changes (reset, superseding turn), cancel the task and count its tokens as wasted.
4. **Reconcile at `utterance_end`** (§Algorithms). Emit a `controller_decision` of tier `T1` with `decision = "COMMIT_RECONCILE"` and `reason = outcome`.
5. **`stands`:** await any unfinished draft futures (emit `predraft_late` note with the wait time). Verify all claims, gate results, ledger v1, emit. TTFT is measured from `utterance_end` to the first `answer_chunk`.
6. **`refined`:** treat the pending claims as a draft ledger. Call the Phase 7 delta planner with the final utterance's tail as the constraint. Retrieve only the ≤ 2 delta queries. Call the rewriter on the affected pending claims. Verify revised and new claims. Verify the untouched pending claims. Write ledger v1 (the pending claims after refinement) and emit `answer_delta` with `change_type = "initial"`. The v1 lineage includes a note `reconciled_from_predraft`.
7. **`redone`:** discard the pending draft (all its tokens become `wasted_tokens`), then run the Phase 5 full path on the final buffer.
8. **Drafts are verified at reconcile time**, not earlier. Verification is pure CPU (milliseconds), and this keeps a single point where claims enter the ledger.
9. **Waste accounting.** At turn end compute:
   - `wasted_tokens` = tokens of discarded pre-drafts plus tokens of any pre-draft sub-intent replaced by redraft,
   - `wasted_provisional` = provisional retrievals whose cache entry was never used,
   - `wasted_speculation_rate` = `wasted_provisional / provisional` (per turn, aggregated by the harness),
   Include wasted tokens in `cost_per_turn`.
10. **Flags.** `speculation.provisional` and `speculation.predraft` independently toggleable. `provisional=false, predraft=false` behaves like B1. `provisional=true, predraft=false` is the Phase 4/5 behavior.
11. **A3 experiment.** Three arms on the test split: `off` (B1-equivalent), `retrieval_only` (provisional + cache, no pre-draft), `full` (provisional + pre-draft + reconcile). Metrics: TTFT p50/p95 (compound and single_early), cost per turn, wasted-speculation rate, cache-hit rate, reconcile-outcome distribution, groundedness (must not drop).
12. **Add adversarial scenarios** (author 6 more, tune split 3 / test split 3, flagged `category: compound`, `tag: tail_change`): the utterance's last clause changes a slot (for example a different quantity), or adds a new topic after the commit point. These test `refined` and `redone` and feed the failure analysis.

## Algorithms / Logic

**Reconcile decision:**
1. Compute `Q_final` from the final buffer (Phase 4 query builder) and its slots and embedding.
2. `tail_new_anchors` = anchors in the final buffer not present in the committed text's anchors.
3. **Stands** if `tail_new_anchors = ∅` AND `cos(emb(Q_final), committed_emb) ≥ 0.82` AND no slot conflict against `committed_slots`.
4. **Refined** if there are new anchors or a tail phrase that adds an intent, AND no slot conflict, AND the number of affected sub-intents ≤ 2.
5. **Redone** if there is a slot conflict on a shared slot (for example 30 → 50 people), or the delta planner fails/returns `unrelated` for the whole turn, or the pending draft was cancelled.

**Pre-draft validity per sub-intent:** a pre-draft is valid only if its sub-query still matches the final plan by the cache reuse rule (cos ≥ 0.82, no slot conflict). Otherwise that sub-intent is redrafted. This never causes a wrong claim to be emitted, only extra cost, which is counted.

**Where the TTFT gain comes from:** LLM planning, per-sub-intent drafting and gate/retrieval finish during the user's speech. At `utterance_end` only reconcile (a few ms) and verification remain. Retrieval caching itself contributes little on a small corpus. The report states this and shows the arms.

**Ordering guarantee:** claims are emitted in sub-intent order. A sub-intent whose draft is still running at `utterance_end` delays only itself and later sub-intents.

**Consistency guarantee:** pre-drafted claims are never emitted unverified, and never emitted if their evidence was gathered for a superseded query.

## Integration

- Extends the Phase 4 COMMIT stage and the Phase 5 planner and retrieval into a background task.
- Reuses the Phase 7 delta planner, rewriter and apply logic for reconcile refinement.
- Reuses the Phase 3 verifier and emitter. Uses the Phase 6 harness for A3 and waste metrics.
- Phase 9 shows pre-draft readiness and reconcile outcome on the timeline.

## Verification

```bash
pytest -q tests/test_predraft.py tests/test_reconcile.py tests/test_speculation_waste.py
pytest -q -m integration tests/test_ttft_arms.py
for arm in off retrieval_only full; do
  python -m slrag.cli replay eval/scenarios/test --mode ours --speculation $arm --out out/a3_$arm.jsonl --speed 1
done
python -m slrag.cli report out/ --split test --experiment A3
```

Acceptance checks:
- **Reconcile table test:** tail adds nothing → `stands`. Tail adds a new topic → `refined`. Tail changes "30" to "50" → `redone` (and the pre-draft tokens appear in `wasted_tokens`). Cancelled pending draft → `redone`.
- **Ready-before-end:** on the Example 1 timeline (commit at 1.6 s, end at 2.1 s), assert in the trace that at least one `draft_completed` with `speculative: true` occurs before `utterance_end` on hardware where the LLM can finish, and record the fraction of compound test turns where all drafts were ready by `utterance_end`.
- **Late draft:** if a draft is not ready at `utterance_end`, the engine waits for it, emits `predraft_late`, and does not emit an unverified claim.
- **Safety:** an integration test where the tail changes the sub-query confirms no claim from the discarded plan appears in the answer or ledger.
- **A3 measurement:** report TTFT p50/p95 for `off` vs `retrieval_only` vs `full`, plus cost per turn including waste. The claim we can make is whatever the measured difference is on this hardware. If `full` is not faster than `off`, record why (for example, pre-drafts not ready on a CPU-only host).
- **Groundedness parity:** groundedness in `full` is not lower than in `off` by more than 2 points on the test split. Zero fabricated cites.
- **Coverage:** `trace_coverage = 1.0` including reconcile events.

## Demo Capability

The TTFT scene: on the same streamed compound utterance, the recorded B1 trace is still working after `[Utterance End]` while Ours already has drafts ready and starts streaming. The timeline shows PROVISIONAL at 0.8 s, COMMIT and drafts completing at ~1.6–2.0 s, reconcile `stands` at 2.1 s, and the TTFT difference and wasted-speculation counter.

## Definition of Done

- [ ] Pre-draft runs in the background at COMMIT and holds drafts without emitting.
- [ ] Reconcile handles `stands`, `refined` and `redone`, all tested.
- [ ] No unverified or superseded claim is ever emitted.
- [ ] Waste (tokens, provisional retrievals) is counted in cost and reported.
- [ ] A3 three-arm results produced on the test split with honest interpretation.
- [ ] Toggling `speculation.*` cleanly reproduces B1 and Phase 5 behaviors.
- [ ] Phase 1–7 tests still pass. Late-detail (Phase 7) behavior unchanged.

## Failure / Rollback

- No TTFT gain: check whether drafts are ready before `utterance_end` (`predraft_ready_before_end`). If the LLM is too slow, report per-hardware and consider a GPU or a hosted OpenAI-compatible endpoint via `LLM_BASE_URL`. Do not tune scenarios to hide it.
- Frequent `redone`: inspect the reconcile tail rules and the COMMIT timing. Commit later only via config (for example require 2 stable chunks), not by special-casing.
- Inconsistent answers after `refined`: check that only affected pending claims were sent to the rewriter (the Phase 7 scope assertion also applies here).
- Rollback: set `speculation.predraft: false`. The engine returns to Phase 5 behavior with drafting at `utterance_end`, and everything else remains valid.

## Output

- Pending-draft state, background pre-draft pipeline
- Reconcile logic (`stands`/`refined`/`redone`) reusing Phase 7 machinery
- Waste accounting and A3 experiment
- 6 adversarial tail-change scenarios, tests
