# Phase 3 — Grounded Answer Pipeline (LLM, Sufficiency Gate, Drafter, Verifier, Ledger v1)

**Priority:** MUST HAVE

## Objective

Deliver the first end-to-end working prototype: at `utterance_end`, retrieve, decide sufficiency, draft cited claims with the LLM, verify them fail-closed, write ledger version 1, and stream the answer with uncertainty where the corpus is insufficient. This phase builds the grounding guarantees (G4) and the ledger that the later phases mutate. It is deliberately a batch pipeline, which also becomes the core of baseline B1.

## Starting State

- Phases 1–2 complete: retriever, vocab/IDF, contracts, telemetry, gateway, session registry, turn engine skeleton.

## Implementation Scope

**MUST HAVE**
- `llm` service in `docker-compose.yml` (llama.cpp server + Qwen2.5-3B-Instruct Q4_K_M) with pinned model download.
- Async LLM client with JSON-schema constrained decoding and token accounting.
- Sufficiency Gate.
- Claim Drafter (enum-constrained cites).
- Claim Verifier (checks a–e, fail-closed) with feature flag `verifier.enabled`.
- Claim Ledger core (add/list claims, `answer_version = 1`, `text_hash`).
- Answer Emitter (claim-granular `answer_chunk`, `answer_delta`, PDF-shaped `TurnOutput`).
- Batch turn engine mode: on `utterance_end` run the full pipeline with a single query.
- `slrag ask` CLI and `/ws/stream` end-to-end.

**HIGH VALUE**
- Model-fetch script with sha256 verification.

**OPTIONAL**
- Token-count fallback if the server does not return usage.

## Files / Modules

Create/modify:
- `docker-compose.yml`, `scripts/fetch_models.sh`, `models/.gitkeep`
- `src/slrag/llm/client.py`, `schemas.py` (JSON Schemas), `prompts.py`
- `src/slrag/answer/gate.py`, `drafter.py`, `verifier.py`, `emitter.py`
- `src/slrag/state/ledger.py`, `session.py` (holds the ledger)
- `src/slrag/nlp/numbers.py`, `negation.py`, `lemmas.py` (extend)
- `src/slrag/engine/turn_engine.py`, `modes.py` (mode = `batch` for now)
- `tests/test_gate.py`, `test_drafter_schema.py`, `test_verifier.py`, `test_ledger.py`, `test_e2e_batch.py`

## Interfaces & Contracts

```python
class LLMClient:
    async def json_call(self, *, system: str, user: str, schema: dict, max_tokens: int,
                        purpose: str, turn_id: str) -> tuple[dict, Usage]     # temperature 0, seed from config
    async def classify(self, *, system: str, user: str, choices: list[str], purpose: str, turn_id: str) -> tuple[str, Usage]
class Usage(BaseModel): tokens_in:int; tokens_out:int; latency_ms:float

class SufficiencyResult(BaseModel): sub_intent_id:str; answerable:bool; dense_top1:float; coverage:float; reason:str|None

async def draft_claims(sub_intent: SubIntent, evidence: list[Evidence]) -> tuple[list[DraftClaim], Usage]
class DraftClaim(BaseModel): text:str; cites:list[str]

class VerifyResult(BaseModel): ok:bool; failed_checks:list[str]; lexical:float; semantic:float; numbers_ok:bool; negation_ok:bool
def verify_claim(claim: DraftClaim, evidence_by_cite: dict[str, Evidence], turn_evidence_cites: set[str]) -> VerifyResult

class Ledger:
    def add_claims(self, sub_intent_id, claims: list[DraftClaim | VerifiedClaim], version: int) -> list[Claim]
    def active_claims(self) -> list[Claim]
    def render(self) -> str
```

Drafter output JSON Schema:

```json
{"type":"object","properties":{"claims":{"type":"array","maxItems":4,
  "items":{"type":"object","required":["text","cites"],
   "properties":{"text":{"type":"string","maxLength":320},
                 "cites":{"type":"array","minItems":1,"items":{"enum":["<cite strings of this call's chunks>"]}}}}}},
 "required":["claims"]}
```

The `cites` enum is built per call from the chunks passed to the model. This makes an invented Doc ID impossible by construction.

`TurnOutput` is exactly the PDF record (`retrieval_events`, `sub_queries`, `answer`, `citations`, `uncertainty`) plus `meta` (`answer_version`, `retrieval_required`, `reason`, `uncertainty_items`, `ttft_ms`, `tokens_in`, `tokens_out`).

## Implementation Steps

1. **Compose + model.** `docker-compose.yml` with:
   - `llm`: `ghcr.io/ggml-org/llama.cpp:server`, args `-m /models/qwen2.5-3b-instruct-q4_k_m.gguf --parallel 3 --cache-prompt -c 8192 --host 0.0.0.0 --port 8080`, healthcheck on `/health`.
   - `app`: depends on `llm` healthy, env `LLM_BASE_URL`, `LLM_MODEL`.
   - `scripts/fetch_models.sh` downloads the GGUF from a pinned revision URL and checks its sha256, failing on mismatch.
2. **LLM client.** `httpx.AsyncClient`, base URL from config, per-call timeout 30 s. Uses the OpenAI-compatible `/v1/chat/completions` with `response_format` JSON schema (llama.cpp GBNF from schema). `temperature: 0`, `seed: 13`. Read `usage` for tokens. Emit no telemetry itself. Callers emit events with the returned `Usage`. A semaphore of `llm.parallel` limits concurrency.
3. **Prompts.** `prompts.py` holds generic instructions only: answer only from the numbered chunks, one fact per claim, ≤ 40 words per claim, if the chunks do not answer the question return no claims. Few-shot exemplars use a synthetic unrelated domain (for example a fictional gardening club).
4. **Sufficiency gate** (§Algorithms). Uses the retriever's `embed_query`, the vocab IDF, and spaCy lemmas. Emits `sufficiency_check` with both scores.
5. **Drafter.** For one sub-intent, take ≤ 4 chunks, number them `[Doc_12 §2] text`. Build the cite enum from those chunks. Call `json_call`. Trim to `draft.max_claims`. Emit `draft_completed` with token usage.
6. **Verifier** (§Algorithms). Pure CPU. Emit one `verification` event per turn with per-claim results in the payload.
7. **Ledger.** Create `Claim` objects with `claim_id` (`c1`, `c2`, …, per session, monotonically increasing), `text_hash = sha256(text + "|" + "|".join(sorted(cites)))`, `created_version = last_modified_version = 1`, `history=[{version,action:"add",text}]`. Store a session-scoped evidence dict `chunk_id → Evidence`.
8. **Batch engine.** On `utterance_end`, for the final buffer:
   - `Q = buffer` (cleaned by removing filler tokens only).
   - `instrumented_search(Q, trigger="final")`.
   - One sub-intent `q1 = Q`. Gate → draft → verify → keep passing claims.
   - Write ledger v1, then emit.
9. **Emitter.**
   - Emit `answer_chunk` per claim in order: `first_token = true` on the first, `ts_s` = stream time.
   - Emit one `answer_delta` (`change_type: "initial"`, ops = `add` per claim, `rendered_answer` = claims joined with citations `[Doc_12 §2]`).
   - Emit `uncertainty_flag` for an insufficient sub-intent or one with no surviving claims.
   - Emit `answer_emitted` and `turn_summary` (`ttft_ms`, calls, tokens, cost).
   - Build `TurnOutput`. The `uncertainty` string is `"<sub-intent text> could not be verified from the retrieved corpus."` per item, joined.
10. **Verifier flag.** `verifier.enabled=false` skips checks b–e but keeps check (a) and the enum constraint. (Ablation A5 uses this.)
11. **CLI.** `slrag ask "<question>" [--session s1]` runs a batch turn against a live LLM and prints the `TurnOutput`.

## Algorithms / Logic

**Sufficiency (per sub-intent):**
- `dense_top1` = max dense cosine among the fused evidence.
- `coverage` = IDF-weighted fraction of the sub-query's content lemmas found in the union of the lemmas of the top-3 chunks: `Σ idf(l) for l in Q_lemmas ∩ chunk_lemmas / Σ idf(l) for l in Q_lemmas`.
- `answerable ⇔ dense_top1 ≥ 0.55 AND coverage ≥ 0.50`, else `insufficient`. Insufficient sub-intents skip drafting.

**Verifier (per claim, all must pass):**
- (a) Every cite is in this turn's evidence cite set.
- (b) Every number, currency, percentage and date token in the claim appears in the cited chunks. Normalize by stripping `,`, `$`, and trailing `%`. Compare month names case-insensitively.
- (c) Negation parity: find the sentence in the cited chunks with maximum cosine to the claim. The claim's negation-cue presence must equal that sentence's. Cues: `not, no, never, cannot, n't, without, none, neither, nor, non-, unable, prohibited, ineligible`.
- (d) Lexical support: IDF-weighted fraction of the claim's content lemmas present in the cited chunks' lemmas ≥ `verifier.lexical` (0.60).
- (e) Semantic support: max cosine between the claim embedding and any cited-chunk sentence embedding ≥ `verifier.semantic` (0.70).

A claim that fails any check is dropped. No retry in the hot path. If every claim of a sub-intent fails, that sub-intent becomes an uncertainty item with reason `no_verified_claims`.

**Fail-closed rule:** an exception inside a verifier check counts as a failure for that claim, and is logged.

**TTFT:** measured from the `utterance_end` timestamp to the first `answer_chunk`, both on the server monotonic clock. Answer chunks are emitted only after verification.

## Integration

- Uses the Phase 1 retriever, vocab, `embed_query`.
- Uses the Phase 2 bus, contracts, session registry, `instrumented_search`, and replaces the turn engine skeleton with the batch mode.
- The Ledger, Drafter, Verifier, Gate and Emitter are reused unchanged by Phases 4–8.

## Verification

```bash
bash scripts/fetch_models.sh && docker compose up --build -d
curl -s localhost:8000/health                       # {"app":"ok","llm":"ok"}
python -m slrag.cli ask "<a question the corpus answers>" --session s1
python -m slrag.cli ask "<a question about something not in the corpus>" --session s2
pytest -q tests/test_gate.py tests/test_verifier.py tests/test_ledger.py tests/test_drafter_schema.py
pytest -q -m integration tests/test_e2e_batch.py    # requires the llm service
```

Acceptance checks:
- **Verifier table test** (fixture chunk plus 12 claims): supported paraphrase → pass. Wrong number → fail (b). Missing date → fail (b). Negation flipped → fail (c). Off-topic claim → fail (d). Fluent but unsupported claim → fail (d or e). Cite not in the evidence set → fail (a). Expected result table checked in.
- **Enum constraint**: a schema built for cites `["A §1","B §2"]` rejects any other value. Across 30 integration questions, the number of emitted cites outside the turn's evidence = 0.
- **Gate**: an out-of-corpus question yields `insufficient`, no draft call (`llm_calls = 0` for drafting), an `uncertainty_flag`, and `TurnOutput.uncertainty` non-empty.
- **Ledger**: ids are `c1..`, `text_hash` is deterministic, `render()` includes each claim's cites.
- **Emitter**: exactly one `first_token = true`. `turn_summary.ttft_ms` is set. A coverage check on the trace has no missing events.
- **Isolation**: two sessions asking different questions have disjoint ledgers.
- **Corpus-only**: every emitted cite maps to a corpus chunk, asserted in the e2e test.

## Demo Capability

Ask a question and receive a cited, grounded answer with real telemetry. Ask something the corpus cannot answer and get an explicit uncertainty statement with the gate scores. This is a real prototype, and the base for baseline B1.

## Definition of Done

- [ ] `docker compose up --build` starts `llm` and `app`, and `/health` reports both healthy.
- [ ] Model download is pinned and sha256-verified.
- [ ] Gate, drafter constraint, verifier and ledger tests pass. Verifier table matches expected results.
- [ ] End-to-end answer works with citations from the corpus only. Zero cites outside the evidence set on the 30-question run.
- [ ] Unanswerable questions produce uncertainty, not fabricated claims.
- [ ] Verifier flag works (`enabled=false` path tested).
- [ ] Earlier phase tests still pass.

## Failure / Rollback

- Slow or malformed LLM output: check that the JSON schema reached the server, that `--parallel` and the context size are sane, and that `max_tokens` is high enough for the schema.
- Everything "insufficient": inspect the two gate scores on known-answerable questions and adjust thresholds via config, not code.
- Verifier drops valid claims: inspect the failed check per claim in the `verification` payload before changing thresholds. Calibration happens in Phase 6.
- If the LLM service is down, `/health` reports it and `slrag ask` fails with a clear error. The retrieval CLI and `/debug/search` from Phases 1–2 keep working.

## Output

- Compose file with `llm` and `app`, model fetch script
- LLM client, gate, drafter, verifier, ledger, emitter
- Batch turn engine mode, `slrag ask`
- Unit and integration tests, verifier expectation table
