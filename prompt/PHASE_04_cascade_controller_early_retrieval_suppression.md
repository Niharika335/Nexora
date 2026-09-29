# Phase 4 — Cascade Controller, Early Retrieval, Evidence Cache & Suppression

**Priority:** MUST HAVE (T2 LLM tier and eager/llm-only comparison modes: HIGH VALUE)

## Objective

Turn the batch pipeline into a streaming one. Implement the cascade Retrieval Controller (T0 rules, T1 slots and drift, T2 LLM), speculative provisional retrieval, the session evidence cache, the streaming turn state machine, and the Presenter for presentation-only turns. This delivers PDF Example 1 (early retrieval), Example 3 (suppression) and gate G2.

## Starting State

- Phase 3 complete: LLM client, gate, drafter, verifier, ledger, emitter, batch turn engine, session ledger.
- Ledger v1 exists after a content turn, so the Presenter has claims to reformat.

## Implementation Scope

**MUST HAVE**
- T0 presentation rules, T1 slot/drift logic, controller decisions WAIT / PROVISIONAL / COMMIT / SUPPRESS.
- Query builder `Q_t`.
- Session evidence cache with reuse rule.
- Streaming turn state machine with per-turn caps and epochs (stale-result discard).
- Presenter (SUPPRESS path) with post-check and deterministic fallback.
- COMMIT stage in this phase = retrieval for the committed query (cache-aware). Drafting still happens at `utterance_end`.

**HIGH VALUE**
- T2 LLM tier for the ambiguous band.
- Controller modes `eager`, `rules_only`, `llm_only` (used by ablation A2 in Phase 6).

**OPTIONAL**
- None.

## Files / Modules

Create/modify:
- `src/slrag/control/t0_rules.py`, `t1_slots.py`, `t2_llm.py`, `query_builder.py`, `controller.py`
- `src/slrag/retrieval/cache.py`
- `src/slrag/answer/presenter.py`
- `src/slrag/engine/turn_engine.py` (streaming mode), `modes.py`
- `src/slrag/nlp/lemmas.py` (anchors, slots, dangling-tail detection)
- `config.yaml` (controller and cache blocks already defined)
- `tests/test_t0.py`, `test_t1.py`, `test_controller_streams.py`, `test_cache.py`, `test_presenter.py`, `tests/fixtures/streams/*.jsonl`

## Interfaces & Contracts

```python
class Decision(str, Enum): WAIT="WAIT"; PROVISIONAL="PROVISIONAL"; COMMIT="COMMIT"; SUPPRESS="SUPPRESS"

class ControllerResult(BaseModel):
    decision: Decision
    tier: Literal["T0","T1","T2"]
    query: str | None                 # Q_t when PROVISIONAL/COMMIT
    reason: str                       # e.g. "slots_insufficient", "first_slot_ok", "stable_2", "multi_intent_boundary", "presentation_restructure"
    commit_hint: dict                 # {"multi_intent": bool}
    payload: dict                     # slots, drift_cos, t0_score, anchors...

class RetrievalController:
    async def on_chunk(self, turn: TurnState, ledger: Ledger | None) -> ControllerResult
class TurnState: buffer:str; chunks:list; last_dispatched_q:str|None; last_dispatched_emb; n_provisional:int; n_commit:int; stable_count:int; committed_q:str|None; epoch:int

class EvidenceCache:                      # per session
    def lookup(self, query:str, emb, slots:dict) -> CacheHit | None
    def store(self, retrieval_id, query, emb, slots, evidence)
    def mark_used(self, retrieval_id)     # for waste accounting
class Presenter:
    async def render(self, instruction:str, ledger:Ledger) -> PresentationResult   # bullets + claim_ids, no retrieval
```

Presenter LLM schema: `{"bullets":[{"text":str,"claim_ids":[str]}]}` with `claim_ids` enum-constrained to the ledger's active claim IDs.

## Implementation Steps

1. **Query builder** (`query_builder.py`). Build `Q_t` from the buffer: keep NOUN, PROPN, NUM, non-auxiliary VERB and ADJ lemmas plus full entity spans, preserve token order, then strip a trailing dangling tail (preposition, coordinating conjunction, determiner, auxiliary, `…`). Return `Q_t`, its content-token count, and the dangling flag.
2. **Slots** (`t1_slots.py`). Extract anchors from the buffer:
   - entity spans (spaCy NER: GPE, ORG, PERSON, PRODUCT, LOC, EVENT),
   - quantities and dates (`CARDINAL`, `QUANTITY`, `DATE`, `MONEY`, `PERCENT`),
   - domain nouns whose lemma is in `corpus_vocab.has()` and has `idf ≥ median IDF`.
   Return `slots = {type: value}` for cache conflict checks (for example `{"CARDINAL":"30","GPE":"Pune"}`).
3. **T0** (`t0_rules.py`). Implement the score and the routing in §Algorithms. The lexicons are policy vocabulary and are stored in `t0_rules.py` as data constants.
4. **T1** (`t1_slots.py`, `controller.py`). Implement WAIT / PROVISIONAL / COMMIT per §Algorithms. Embeddings come from `retriever.embed_query`, computed in a thread executor.
5. **T2** (`t2_llm.py`). `classify(choices=["RETRIEVE","NO_RETRIEVE"])` with a 400 ms timeout. On timeout or error return `RETRIEVE`, tier `T2`, and reason `t2_timeout`. Only called in the ambiguous band.
6. **Modes.** `controller.mode` ∈ `cascade` (default), `rules_only` (T2 skipped; ambiguous band → continue to T1), `llm_only` (T2 on every chunk), `eager` (PROVISIONAL retrieval on every chunk, no controller logic), `batch` (Phase 3 behavior).
7. **Cache** (`cache.py`). Entries store `{retrieval_id, query, emb, slots, evidence, used:bool}`. `lookup` returns the entry with the highest cosine that satisfies the reuse rule. Emit `cache_lookup` events (hit/miss, cosine, conflict).
8. **Streaming turn state machine** (`turn_engine.py`).
   - Per turn: `TurnState`, an `epoch` counter, and `asyncio.Task` handles for outstanding retrievals.
   - On chunk: run the controller under the session lock, then act:
     - WAIT → nothing.
     - PROVISIONAL → dispatch `instrumented_search(Q_t, trigger="provisional")`, store in cache on completion.
     - COMMIT → dispatch retrieval for `Q_t` (`trigger` = `multi_intent` if `commit_hint.multi_intent` else `final`), reuse the cache if the rule holds (`source: "cache"`), set `committed_q`.
     - SUPPRESS → call the Presenter path.
   - Every dispatched task carries `(turn_id, epoch, retrieval_id)`. If the epoch changed when the task finishes, discard its result and log `stale_discard`.
   - On `utterance_end`: final path. If `committed_q` exists and the final query matches by the cache rule, reuse the committed evidence. Otherwise retrieve fresh. Then run the Phase 3 gate → draft → verify → ledger → emit. If nothing committed, use the provisional cache entry if it matches, else fresh.
9. **Presenter** (§Algorithms). Emits `suppression` (reason), `answer_delta` with `change_type: "presentation"` (same `answer_version`, a `render_id`), `answer_emitted`, `turn_summary` with `retrieval_calls = 0`.
10. **Presentation-only turns carry no retrieval events.** `TurnOutput.retrieval_events = []`, `meta.retrieval_required = false`, `meta.reason = "presentation_restructure"`.
11. **Tests and fixtures.** Add scripted streams in `tests/fixtures/streams/` covering: dangling tail, first slot_ok, drift change, stable commit, multi-intent boundary, presentation-only, presentation-like-but-content (new anchors), ambiguous.

## Algorithms / Logic

**T0 score** (evaluated only if the session already has a ledger with ≥ 1 active claim):
- `verb` = 1 if a presentation verb is present (`repeat, shorten, summarize, summarise, rephrase, reword, translate, bullet(s), simplify, reformat, condense, restate, tl;dr, shorter, briefly`), else 0.
- `anaph` = 1 if a reference to prior output is present (`that, this, it, those, them, above, previous, last answer, your answer, what you said, earlier`), else 0.
- `noanchor` = 1 if the utterance has **no new anchors**, meaning no entity, quantity or domain noun that is absent from the ledger vocabulary (sub-intent texts and active claim texts), else 0.
- `score = 0.5·verb + 0.3·anaph + 0.2·noanchor`.
- `score ≥ 0.70 AND noanchor = 1` → **SUPPRESS** (all four frozen conditions hold).
- `score < 0.40` → continue to T1.
- Otherwise (0.40 ≤ score < 0.70, or score ≥ 0.70 with new anchors) → T2 asks RETRIEVE/NO_RETRIEVE. NO_RETRIEVE → SUPPRESS. RETRIEVE → continue to T1.

**T1:**
- `content_tokens(Q_t) < 4` → WAIT.
- `anchors < 2` and not (`anchors == 1` and head verb present and `content_tokens ≥ 6`) → WAIT.
- Otherwise the slots are sufficient:
  - **PROVISIONAL** if no retrieval has been dispatched this turn, or `cos(emb(Q_t), last_dispatched_emb) < 0.82` (new information), and `n_provisional < 2`.
  - **COMMIT** (`n_commit < 1`) if either
    - the multi-intent boundary rule holds: ≥ 2 distinct topic noun-phrases in `Q_t` joined by a coordinator or comma, no dangling tail, ≥ 2 anchors → `commit_hint.multi_intent = true`; or
    - `Q_t` has been stable for 2 consecutive chunks after a PROVISIONAL: `cos(Q_t, last_Q) ≥ 0.88` and no dangling tail.
  - else WAIT.
- `utterance_end` always forces the final path if there was no COMMIT.

**Cache reuse rule:** reuse iff `cos(query_emb, entry_emb) ≥ 0.82` and there is no slot conflict (same slot type with different value, for example `CARDINAL: 30` vs `50`).

**Presenter:**
1. Parse the requested bullet count from the instruction (digits or number words up to ten). Default 3.
2. LLM call over the ledger's active claims only, `claim_ids` enum-constrained. Must reproduce the citations of the source claims.
3. **Post-check:** the set of cites in the output ⊆ union of source claim cites. Every number in the output ⊆ numbers in the source claims. Every NER entity ⊆ entities in the source claims.
4. On failure, use the **deterministic fallback**: take the requested number of active claims in order, truncate each to its first sentence, keep its cites. Log `presenter_fallback`.
5. No retrieval or corpus access occurs in this path.

## Integration

- Uses the Phase 1 retriever (`embed_query`, vocab) and the Phase 2 bus, gateway and cache-facing telemetry.
- Reuses the Phase 3 gate, drafter, verifier, ledger and emitter at `utterance_end`.
- The COMMIT hook and `commit_hint.multi_intent` are consumed by the Phase 5 planner. The cache is reused by Phases 5, 7 and 8.

## Verification

```bash
pytest -q tests/test_t0.py tests/test_t1.py tests/test_controller_streams.py tests/test_cache.py tests/test_presenter.py
docker compose up -d && python -m slrag.cli simulate tests/fixtures/streams/early_retrieval.jsonl --url ws://localhost:8000/ws/stream
python -m slrag.cli trace out/telemetry.jsonl --turn t1     # prints decision per chunk with tier and reason
```

Acceptance checks:
- **Decision table test** (chunk sequences, expected decisions):
  - "I need to plan a customer workshop in…" → WAIT (dangling tail, 1 anchor).
  - "…<city> for 30 people, and I need…" → PROVISIONAL, query contains the city and the number.
  - "…the cancellation policy and the catering options." → COMMIT (`multi_intent_boundary`).
  - A stream of only fillers ("um so", "well the") → WAIT throughout, 0 retrievals.
- **G2 check on fixtures**: `retrieval_started.ts` < `utterance_end.ts` for every eligible fixture stream.
- **False-trigger check**: a fixture set of no-retrieval streams (fillers and pure presentation requests) produces 0 provisional retrievals.
- **Cache test**: same query twice → second `source: "cache"`. `30` vs `50` people → miss (slot conflict). Query with cos 0.80 → miss, 0.84 → hit.
- **Suppression e2e**: run a content turn, then "Please repeat your last answer in two bullets." Expect `retrieval_events == []`, exactly 2 bullets, cites ⊆ previous cites, `answer_version` unchanged, `meta.reason == "presentation_restructure"`, `retrieval_calls == 0`.
- **Presenter post-check test**: a presenter output containing an unseen number triggers `presenter_fallback`.
- **Content-with-verb test**: "Summarize the policy for international trips" (a new anchor) is not suppressed and goes to T1/T2.
- **Stale discard**: bump the epoch during a slow retrieval → result discarded and `stale_discard` logged.
- **Modes**: `eager` produces a provisional retrieval for every chunk on the fixtures, which A2 later measures.

## Demo Capability

The streaming behavior in Examples 1 and 3: the controller lane shows WAIT → PROVISIONAL → COMMIT per chunk, `retrieval_started` fires before `utterance_end`, and "repeat that in two bullets" runs with zero retrievals and no new citations.

## Definition of Done

- [ ] Controller decisions match the fixture table, including WAIT on noise.
- [ ] Early retrieval occurs before `utterance_end` on all eligible fixture streams.
- [ ] Presentation-only turns issue 0 retrievals and invent no citations.
- [ ] Cache reuse rule and slot-conflict tests pass. Stale results are discarded.
- [ ] T2 timeout falls back to RETRIEVE. `rules_only`, `llm_only`, `eager`, `batch` modes selectable by config.
- [ ] Coverage check passes for both content and suppressed turns.
- [ ] Phase 1–3 tests and `slrag ask` (batch mode) still pass.

## Failure / Rollback

- Too many false triggers: print the slot values and content-token counts per chunk from the `controller_decision` payload. Adjust `min_content_tokens` and the anchor rule via config, not with special cases.
- No early retrieval: check the anchor detector on the corpus vocabulary. Anchors from `has()` require the corpus to contain the noun.
- Presenter invents facts: the post-check must fail closed to the deterministic fallback.
- If streaming mode is unstable, set `controller.mode: batch` to return to the Phase 3 behavior. The batch pipeline stays available.

## Output

- T0/T1/T2 controller, query builder, slot extraction
- Evidence cache, streaming turn state machine with epochs
- Presenter with post-check and fallback
- Controller mode flags, scripted stream fixtures, tests
