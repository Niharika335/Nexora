"""Replay metrics reduced from a telemetry event stream (dicts or event models)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

REQUIRED_METRICS = (
    "recall@5",
    "recall@10",
    "early_retrieval_rate",
    "false_trigger_rate",
    "subintent_recall",
    "over_fragmentation",
    "suppression_precision",
    "suppression_recall",
    "uncertainty_precision",
    "uncertainty_recall",
    "ttft_p50",
    "ttft_p95",
    "cost_per_turn",
    "wasted_speculation_rate",
    "cache_hit_rate",
    "trace_coverage",
    "hallucinated_id_rate",
)

# Phase 7 metrics: None when the replayed scenarios contain no late-detail (refinement) turn.
# Phase 8 speculation metrics: None when no commit-stage pre-draft ran.
SPECULATION_METRICS = (
    "predraft_turns",
    "reconcile_stands",
    "reconcile_refined",
    "reconcile_redone",
    "wasted_tokens",
    "predraft_tokens",
    "predraft_waste_rate",
    "predraft_ready_before_end_rate",
)

LATE_DETAIL_METRICS = (
    "refinement_turns",
    "preservation_rate",
    "version_lineage_complete",
    "late_restart_turns",
    "contradiction_guard_flags",
    "contradiction_rate_auto",
    "refinement_retrieval_calls",
    "refinement_tokens",
)


def calculate_percentile(values: List[float], p: float) -> float:
    """Deterministic linear-interpolated percentile (numpy method='linear')."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    if n == 1:
        return sorted_v[0]
    rank = (n - 1) * p
    k = int(rank)
    d = rank - k
    if k + 1 < n:
        return (1.0 - d) * sorted_v[k] + d * sorted_v[k + 1]
    return sorted_v[k]


def _as_dict(event: Any) -> Dict[str, Any]:
    if isinstance(event, dict):
        return event
    if hasattr(event, "model_dump"):
        return event.model_dump(mode="json")
    return dict(getattr(event, "__dict__", {}))


def _ratio(num: float, den: float, empty: float) -> float:
    return num / den if den > 0 else empty


class MetricCalculator:
    """Computes the Phase 6 metric set.

    Definitions:
      recall@k               per turn: |gold ∩ ⋃ top-k retrieved over the turn's sub-queries| / |gold|, averaged
      early_retrieval_rate   early (pre-utterance_end) retrievals / all retrievals
      false_trigger_rate     early retrievals on turns that needed none / all retrievals
      wasted_speculation_rate  early retrievals whose evidence was never used (wasted, invalidated, stale) / all retrievals
      subintent_recall       Σ min(resolved, gold sub-intents) / Σ gold (resolved / planned when no gold)
      over_fragmentation     turns planned into more sub-queries than gold sub-intents / turns with gold
      suppression_*          sub-intents suppressed vs unanswerable ground truth
      uncertainty_*          sub-intents flagged uncertain vs unanswerable ground truth
      ttft_p50/p95 (ms)      first answer token - utterance_end (turn_start if no utterance_end)
      cost_per_turn          mean est_cost_usd over completed turns
      cache_hit_rate         cache hits / cache lookups (hit + miss)
      hallucinated_id_rate   emitted citations outside their sub-intent's evidence / emitted citations
      trace_coverage         mean per-turn fraction of the stages required for that turn's kind:
                             content turns need all canonical stages; insufficient turns (every
                             sub-intent suppressed, nothing drafted) need no verification;
                             presentation turns (suppression event) need no retrieval,
                             sufficiency or verification; refinement turns (plan_completed) also
                             need answer_version_transition, and a `modifies` turn needs drafting
                             and verification only when it revised or added a claim

    Retrieval-rate metrics (early / false trigger / wasted, i.e. gate G2) exclude refinement turns
    (turns with a plan_completed event) entirely: on those turns early retrieval is paused until
    the delta plan at utterance_end by design, and they are judged by G5 instead. Their retrievals
    (delta or, after an adds fallback, Phase 5) are reported separately as `refinement_retrievals`.

    Phase 8 (speculation) metrics, None when no pre-draft ran: predraft_turns, reconcile_{stands,refined,
    redone} (turn counts by outcome), wasted_tokens / predraft_tokens (sums; wasted = tokens of discarded
    pre-draft LLM calls, also included in cost_per_turn via est_cost_usd), predraft_waste_rate
    (wasted / pre-draft tokens) and predraft_ready_before_end_rate. A turn with a speculative
    draft_completed also needs its reconcile_completed event for trace coverage.

    Phase 7 (late-detail) metrics, None when no refinement turn was replayed:
      refinement_turns          turns routed through the delta planner
      preservation_rate         unaffected + kept claims byte-identical after refinement / all of them
      version_lineage_complete  answer_version_transition steps that continue the previous version / all steps
      late_restart_turns        refinement turns that restarted or re-searched outside the delta queries
      contradiction_guard_flags revised/new claims downgraded by the contradiction guard
      contradiction_rate_auto   refinement turns whose final live claims still contain a guard-detectable
                                (noun, number) conflict / refinement turns. Automatic proxy; the spec's
                                contradiction_rate is human-labeled and is not computed here
      refinement_retrieval_calls, refinement_tokens   mean per refinement turn (tokens in + out)
    """

    CANONICAL_TURN_STAGES = {"turn_start", "retrieval", "sufficiency", "drafting", "verification", "turn_complete"}
    REFINEMENT_BASE_STAGES = {"turn_start", "retrieval", "sufficiency", "turn_complete", "version_transition"}
    INSUFFICIENT_TURN_STAGES = CANONICAL_TURN_STAGES - {"verification"}
    PRESENTATION_TURN_STAGES = {"turn_start", "drafting", "turn_complete"}

    @staticmethod
    def calculate_all(events: List[Any]) -> Dict[str, Optional[float]]:
        turn_start: Dict[str, float] = {}
        utterance_end: Dict[str, float] = {}
        first_output: Dict[str, float] = {}
        turn_stages: Dict[str, Set[str]] = {}

        cache_hits = cache_lookups = 0
        spec_total = spec_early = spec_wasted = false_triggers = 0
        planned_total = resolved_total = 0
        gold_total = resolved_vs_gold = 0
        frag_turns = frag_over = 0
        supp_tp = supp_fp = supp_fn = 0
        unc_tp = unc_fp = unc_fn = 0
        costs: List[float] = []
        emitted_cites = hallucinated_cites = 0
        presentation_turns: Set[str] = set()
        predraft_turns: Set[str] = set()
        reconciles: List[Dict[str, Any]] = []
        completions: Dict[str, List[bool]] = {}
        groundedness: List[float] = []
        recall_gold: Dict[str, Set[str]] = {}
        recall_hits: Dict[int, Dict[str, Set[str]]] = {5: {}, 10: {}}
        plan_relation: Dict[str, str] = {}
        transitions: Dict[str, List[Dict[str, Any]]] = {}
        restart_turns: Set[str] = set()
        non_delta_retrieval_turns: Set[str] = set()
        refinement_retrievals = 0
        turn_cost: Dict[str, Dict[str, float]] = {}

        dicts = [_as_dict(raw) for raw in events]
        refinement_turn_ids = {e.get("turn_id") or "" for e in dicts if e.get("event_type") == "plan_completed"}

        for e in dicts:
            etype = e.get("event_type")
            tid = e.get("turn_id") or ""
            ts = e.get("timestamp", 0.0)
            ts = float(ts) if isinstance(ts, (int, float)) else 0.0
            stages = turn_stages.setdefault(tid, set()) if tid else set()

            if etype == "turn_start":
                stages.add("turn_start")
                turn_start[tid] = ts
            elif etype == "utterance_final":
                utterance_end[tid] = ts
            elif etype in ("first_token_emission", "first_output_emission"):
                stages.add("drafting")
                first_output.setdefault(tid, ts)
            elif etype == "turn_complete":
                stages.add("turn_complete")
                if "est_cost_usd" in e:
                    costs.append(float(e["est_cost_usd"]))
                emitted_cites += int(e.get("emitted_cites", 0) or 0)
                hallucinated_cites += int(e.get("hallucinated_cites", 0) or 0)
                turn_cost[tid] = {
                    "calls": float(e.get("retrieval_calls", 0) or 0),
                    "tokens": float((e.get("tokens_in", 0) or 0) + (e.get("tokens_out", 0) or 0)),
                }
            elif etype == "suppression":
                presentation_turns.add(tid)
            elif etype == "speculative_retrieval":
                stages.add("retrieval")
                if e.get("trigger") != "refinement":
                    non_delta_retrieval_turns.add(tid)
                if tid in refinement_turn_ids or e.get("trigger") == "refinement":
                    refinement_retrievals += 1
                    continue
                spec_total += 1
                if e.get("is_early"):
                    spec_early += 1
            elif etype == "speculative_cache":
                if e.get("action") in ("hit", "miss"):
                    cache_lookups += 1
                    cache_hits += e.get("action") == "hit"
            elif etype == "speculation_outcome" and tid not in refinement_turn_ids:
                outcome = e.get("outcome")
                if outcome in ("wasted", "invalidated", "stale_discard"):
                    spec_wasted += 1
                elif outcome == "false_trigger":
                    false_triggers += 1
            elif etype == "sub_intent_retrieval":
                stages.add("retrieval")
                expected = set(e.get("expected_chunk_ids") or [])
                if expected:
                    recall_gold.setdefault(tid, set()).update(expected)
                    retrieved = e.get("retrieved_chunk_ids") or []
                    for k in (5, 10):
                        recall_hits[k].setdefault(tid, set()).update(retrieved[:k])
            elif etype == "sub_intent_completion":
                stages.add("sufficiency")
                completions.setdefault(tid, []).append(bool(e.get("is_suppressed")))
                gt = e.get("is_unanswerable_ground_truth")
                if gt is not None:
                    is_supp, is_unc = bool(e.get("is_suppressed")), bool(e.get("is_uncertain"))
                    supp_tp += is_supp and gt
                    supp_fp += is_supp and not gt
                    supp_fn += (not is_supp) and gt
                    unc_tp += is_unc and gt
                    unc_fp += is_unc and not gt
                    unc_fn += (not is_unc) and gt
            elif etype == "multi_intent_resolution":
                planned = int(e.get("total_sub_intents", 0))
                resolved = int(e.get("resolved_count", 0))
                planned_total += planned
                resolved_total += resolved
                gold = e.get("gold_sub_intents")
                if gold:
                    gold_total += int(gold)
                    resolved_vs_gold += min(resolved, int(gold))
                    frag_turns += 1
                    frag_over += planned > int(gold)
            elif etype == "verification":
                stages.add("verification")
                groundedness.append(float(e.get("groundedness_score", 1.0)))
            elif etype == "plan_completed":
                plan_relation[tid] = (e.get("payload") or {}).get("relation", "")
            elif etype == "answer_version_transition":
                stages.add("version_transition")
                transitions.setdefault(tid, []).append(e.get("payload") or {})
            elif etype == "draft_completed" and e.get("speculative"):
                predraft_turns.add(tid)
            elif etype == "reconcile_completed":
                stages.add("reconcile")
                reconciles.append(e)
            elif etype == "reconciliation" and e.get("reconciliation_type") == "restart":
                restart_turns.add(tid)

        ttft = [
            max(0.0, (t_first - utterance_end.get(tid, turn_start[tid])) * 1000.0)
            for tid, t_first in first_output.items()
            if tid in utterance_end or tid in turn_start
        ]

        def recall_at(k: int) -> float:
            scores = [len(g & recall_hits[k].get(tid, set())) / len(g) for tid, g in recall_gold.items()]
            return sum(scores) / len(scores) if scores else 1.0

        def changed(tid: str) -> bool:
            return any(p.get("revised") or p.get("added") for p in transitions.get(tid, []))

        def required(tid: str) -> Set[str]:
            base = required_base(tid)
            return base | {"reconcile"} if tid in predraft_turns else base

        def required_base(tid: str) -> Set[str]:
            if tid in presentation_turns:
                return MetricCalculator.PRESENTATION_TURN_STAGES
            if plan_relation.get(tid) == "modifies":
                extra = {"drafting", "verification"} if changed(tid) else set()
                return MetricCalculator.REFINEMENT_BASE_STAGES | extra
            if tid in plan_relation:  # adds / unrelated: the Phase 5 flow plus the version transition
                base = (MetricCalculator.INSUFFICIENT_TURN_STAGES if completions.get(tid) and all(completions[tid])
                        else MetricCalculator.CANONICAL_TURN_STAGES)
                return base | {"version_transition"}
            if completions.get(tid) and all(completions[tid]):
                return MetricCalculator.INSUFFICIENT_TURN_STAGES
            return MetricCalculator.CANONICAL_TURN_STAGES

        coverages = [len(s & required(tid)) / len(required(tid)) for tid, s in turn_stages.items()]

        spec: Dict[str, Optional[float]] = {k: None for k in SPECULATION_METRICS}
        spec["predraft_turns"] = float(len(predraft_turns | {r.get("turn_id") or "" for r in reconciles}))
        if reconciles:
            for outcome in ("stands", "refined", "redone"):
                spec[f"reconcile_{outcome}"] = float(sum(1 for r in reconciles if r.get("outcome") == outcome))
            wasted = float(sum(int(r.get("wasted_tokens", 0) or 0) for r in reconciles))
            spent = float(sum(int(r.get("predraft_tokens", 0) or 0) for r in reconciles))
            spec.update(wasted_tokens=wasted, predraft_tokens=spent, predraft_waste_rate=wasted / spent if spent else 0.0,
                        predraft_ready_before_end_rate=sum(1 for r in reconciles if r.get("predraft_ready_before_end")) / len(reconciles))

        refinement = sorted(plan_relation)
        late: Dict[str, Optional[float]] = {k: None for k in LATE_DETAIL_METRICS}
        late["refinement_turns"] = float(len(refinement))
        late["refinement_retrievals"] = float(refinement_retrievals)
        if refinement:
            ref_payloads = [p for tid in refinement for p in transitions.get(tid, [])]
            checked = sum(int(p.get("preserved_checked", 0) or 0) for p in ref_payloads)
            identical = sum(int(p.get("preserved_identical", 0) or 0) for p in ref_payloads)
            late["preservation_rate"] = identical / checked if checked else 1.0
            late["late_restart_turns"] = float(sum(
                1 for tid in refinement
                if tid in restart_turns or (plan_relation[tid] == "modifies" and tid in non_delta_retrieval_turns)
            ))
            late["contradiction_guard_flags"] = float(sum(len(p.get("contradictions") or []) for p in ref_payloads))
            late["contradiction_rate_auto"] = sum(
                1 for tid in refinement if any((p.get("post_refinement_conflicts") or 0) > 0 for p in transitions.get(tid, []))
            ) / len(refinement)
            costs_ref = [turn_cost[tid] for tid in refinement if tid in turn_cost]
            late["refinement_retrieval_calls"] = sum(c["calls"] for c in costs_ref) / len(costs_ref) if costs_ref else None
            late["refinement_tokens"] = sum(c["tokens"] for c in costs_ref) / len(costs_ref) if costs_ref else None
        all_transitions = [p for ps in transitions.values() for p in ps]
        if all_transitions:
            late["version_lineage_complete"] = sum(1 for p in all_transitions if p.get("lineage_ok", True)) / len(all_transitions)

        if gold_total:
            subintent_recall = resolved_vs_gold / gold_total
        else:
            subintent_recall = _ratio(resolved_total, planned_total, 1.0)

        return {
            "recall@5": recall_at(5),
            "recall@10": recall_at(10),
            "early_retrieval_rate": _ratio(spec_early, spec_total, 0.0),
            "false_trigger_rate": _ratio(false_triggers, spec_total, 0.0),
            "wasted_speculation_rate": _ratio(spec_wasted, spec_total, 0.0),
            "subintent_recall": subintent_recall,
            "sub_intent_recall": subintent_recall,  # alias used by calibration and older reports
            "over_fragmentation": _ratio(frag_over, frag_turns, 0.0),
            "suppression_precision": _ratio(supp_tp, supp_tp + supp_fp, 1.0),
            "suppression_recall": _ratio(supp_tp, supp_tp + supp_fn, 1.0),
            "uncertainty_precision": _ratio(unc_tp, unc_tp + unc_fp, 1.0),
            "uncertainty_recall": _ratio(unc_tp, unc_tp + unc_fn, 1.0),
            "ttft_p50": calculate_percentile(ttft, 0.50),
            "ttft_p95": calculate_percentile(ttft, 0.95),
            "cost_per_turn": sum(costs) / len(costs) if costs else None,
            "cache_hit_rate": _ratio(cache_hits, cache_lookups, 0.0),
            "hallucinated_id_rate": _ratio(hallucinated_cites, emitted_cites, 0.0),
            "trace_coverage": sum(coverages) / len(coverages) if coverages else 0.0,
            "groundedness": sum(groundedness) / len(groundedness) if groundedness else None,
            **late,
            **spec,
        }
