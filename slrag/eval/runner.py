"""Deterministic replay runner executing B0, B1 and ours on identical pipeline code."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from slrag.config import DEFAULT_CONFIG
from slrag.eval.clock import DEFAULT_LATENCY, ReplayClock
from slrag.eval.gates import GateEvaluator
from slrag.eval.report import ReportGenerator
from slrag.eval.scenarios import write_split_files
from slrag.replay.baselines import DEFAULT_CORPORA, MODE_FLAGS, build_index, build_turn_engine, mode_config
from slrag.replay.metrics import MetricCalculator


class InMemoryBus:
    def __init__(self):
        self.events: List[Any] = []

    def publish(self, event: Any) -> None:
        self.events.append(event)


INTER_TURN_GAP_S = 2.0  # virtual seconds between the two turns of a late-detail session
A3_ARMS = ("off", "retrieval_only", "full")
A3_METRICS = (
    "ttft_p50", "ttft_p95", "cost_per_turn", "wasted_speculation_rate", "cache_hit_rate", "early_retrieval_rate",
    "groundedness", "hallucinated_id_rate", "trace_coverage", "predraft_turns", "reconcile_stands", "reconcile_refined",
    "reconcile_redone", "wasted_tokens", "predraft_tokens", "predraft_waste_rate", "predraft_ready_before_end_rate",
)


def load_scenarios(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_scenario_files(path: str | Path, split: str = "test") -> List[Path]:
    """A scenario file, a scenario directory (-> <split>.jsonl + late_<split>.jsonl), or
    <dir>/test | <dir>/tune naming a split inside a scenario directory."""
    p = Path(path)
    if p.is_file():
        return [p]
    if not p.exists() and p.name in ("test", "tune") and p.parent.is_dir():
        p, split = p.parent, p.name
    if p.is_dir():
        return [f for f in (p / f"{split}.jsonl", p / f"late_{split}.jsonl") if f.exists()]
    return [p]


def turn_ids(sc: Dict[str, Any]) -> List[str]:
    """Turn ids of a scenario: the scenario id, or <id>:t1 / <id>:t2 for a two-turn late-detail session."""
    return [f"{sc['scenario_id']}:t1", f"{sc['scenario_id']}:t2"] if sc.get("follow_up") else [sc["scenario_id"]]


def _event_dict(event: Any) -> Dict[str, Any]:
    if isinstance(event, dict):
        return event
    if hasattr(event, "model_dump"):
        return event.model_dump(mode="json")
    return dict(event.__dict__)


class ReplayRunner:
    """Replays scenario files through the TurnEngine in `mode` (ours | b1 | b0).

    Each scenario is its own session (fresh engine, ledger and cache), streamed in word chunks
    on a deterministic virtual clock. `speed` > 0 additionally paces wall-clock time (1.0 = real time);
    metrics always use virtual time, so results are identical at any speed.
    """

    def __init__(
        self,
        scenarios_path: str,
        mode: str = "ours",
        out_path: str = "out/telemetry.jsonl",
        playback: str = "virtual",
        config_overrides: Optional[Dict[str, Any]] = None,
        speed: Optional[float] = None,
        corpus_paths: Iterable[Path] = DEFAULT_CORPORA,
        split: str = "test",
        category: Optional[str] = None,
        trace_out: Optional[str] = None,
    ):
        self.scenarios_path = scenarios_path
        self.trace_out = trace_out  # Phase 9: per-scenario trace JSONL (<trace_out>/<mode>_<scenario_id>.jsonl)
        self.split = split
        self.category = category
        self.mode = mode.lower()
        if self.mode not in MODE_FLAGS:
            raise ValueError(f"Unknown mode '{mode}'. Expected one of {sorted(MODE_FLAGS)}.")
        self.out_path = out_path
        self.speed = speed if speed is not None else (1.0 if playback == "real" else 0.0)
        self.config_overrides = config_overrides or {}
        self.corpus_paths = list(corpus_paths)
        self.clock = ReplayClock(start_time=1000.0, tick_interval=0.050)
        self.bus = InMemoryBus()
        self._indexes: Dict[str, Any] = {}

    def _index_for(self, mode: str) -> Any:
        if mode not in self._indexes:
            self._indexes[mode] = build_index(mode_config(mode, self.config_overrides).app, self.corpus_paths)
        return self._indexes[mode]

    def _execute_scenarios(self, mode: str, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.bus.events.clear()
        self.clock.reset(1000.0)
        index = self._index_for(mode)

        for sc in scenarios:
            engine = build_turn_engine(mode, index, self.bus, clock=self.clock, overrides=self.config_overrides)
            first_event = len(self.bus.events)
            sub_gold = {s["id"]: s.get("gold_chunk_ids", []) for s in sc.get("sub_intents", [])}
            t_before = self.clock.time()
            ids = turn_ids(sc)
            asyncio.run(engine.execute_turn(
                query=sc["query"],
                turn_id=ids[0],
                is_unanswerable_ground_truth=sc.get("is_unanswerable", False),
                sub_gold_map=sub_gold,
                late_detail=sc.get("late_detail"),
            ))
            follow_up = sc.get("follow_up")
            if follow_up:
                # Turn 2 of the same session: ours refines the ledger; restart baselines (B0/B1)
                # discard the first pass and re-run retrieval + drafting on the combined request.
                self.clock.advance(INTER_TURN_GAP_S)
                if engine.config.restart_on_late_detail:
                    asyncio.run(engine.execute_turn(query=sc["query"], turn_id=ids[1], late_detail=follow_up["text"]))
                else:
                    asyncio.run(engine.execute_turn(
                        query=follow_up["text"], turn_id=ids[1], is_unanswerable_ground_truth=False,
                        expected_chunk_ids=follow_up.get("delta_gold_chunk_ids", []),
                    ))
            if self.trace_out:
                trace = Path(self.trace_out) / f"{mode}_{sc['scenario_id']}.jsonl"
                trace.parent.mkdir(parents=True, exist_ok=True)
                with open(trace, "w", encoding="utf-8") as f:
                    for ev in self.bus.events[first_event:]:
                        f.write(json.dumps(_event_dict(ev), ensure_ascii=False) + "\n")
            if self.speed > 0:
                time.sleep((self.clock.time() - t_before) / self.speed)
            self.clock.advance(1.0)  # gap between sessions

        return [_event_dict(e) for e in self.bus.events]

    def _scenario_counts(self) -> Dict[str, int]:
        files = resolve_scenario_files(self.scenarios_path, self.split)
        base = files[0].parent if files else Path(self.scenarios_path).parent
        counts = {}
        for split in ("tune", "test"):
            p = base / f"{split}.jsonl"
            counts[split] = len(load_scenarios(p)) if p.exists() else 0
        counts["total"] = counts["tune"] + counts["test"]
        for split in ("tune", "test"):  # Phase 7 late-detail sessions, counted separately
            p = base / f"late_{split}.jsonl"
            counts[f"late_{split}"] = len(load_scenarios(p)) if p.exists() else 0
        return counts

    def load(self) -> List[Dict[str, Any]]:
        """Scenarios from the resolved file(s), filtered by category when one is given."""
        scenarios = [sc for f in resolve_scenario_files(self.scenarios_path, self.split) for sc in load_scenarios(f)]
        if self.category:
            wanted = self.category.replace("-", "_")
            scenarios = [sc for sc in scenarios if sc.get("category", "").replace("-", "_") == wanted]
        return scenarios

    def a4_experiment(self, scenarios: List[Dict[str, Any]], ours_events: List[Dict[str, Any]],
                      restart_events: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        """A4: turn-2 retrieval calls and tokens of refinement (ours) vs restart (B1 with restart on
        late detail), measured on the same late-detail sessions; plus relation / affected accuracy
        of the delta planner against the gold labels."""
        late = [sc for sc in scenarios if sc.get("follow_up")]
        if not late:
            return None
        if restart_events is None:
            restart_events = self._execute_scenarios("b1", late)

        def by_turn(events: List[Dict[str, Any]], etype: str) -> Dict[str, List[Dict[str, Any]]]:
            out: Dict[str, List[Dict[str, Any]]] = {}
            for e in events:
                if e.get("event_type") == etype:
                    out.setdefault(e.get("turn_id", ""), []).append(e)
            return out

        ours_done, rest_done = by_turn(ours_events, "turn_complete"), by_turn(restart_events, "turn_complete")
        plans = by_turn(ours_events, "plan_completed")
        retrievals = by_turn(ours_events, "sub_intent_retrieval")
        rows = []
        for sc in late:
            t2 = turn_ids(sc)[1]
            ours = (ours_done.get(t2) or [{}])[-1]
            rest = (rest_done.get(t2) or [{}])[-1]
            plan = ((plans.get(t2) or [{}])[-1]).get("payload") or {}
            gold = sc["follow_up"]
            retrieved = {c for e in retrievals.get(t2, []) for c in (e.get("retrieved_chunk_ids") or [])[:5]}
            delta_gold = set(gold.get("delta_gold_chunk_ids") or [])
            rows.append({
                "scenario_id": sc["scenario_id"],
                "ours_retrieval_calls": ours.get("retrieval_calls", 0),
                "restart_retrieval_calls": rest.get("retrieval_calls", 0),
                "ours_tokens": (ours.get("tokens_in", 0) or 0) + (ours.get("tokens_out", 0) or 0),
                "restart_tokens": (rest.get("tokens_in", 0) or 0) + (rest.get("tokens_out", 0) or 0),
                "relation": plan.get("relation"), "gold_relation": gold.get("relation"),
                "affected": plan.get("affected_sub_intents"), "gold_affected": gold.get("affected_sub_intents"),
                "fallback": plan.get("fallback"),
                "delta_recall@5": len(delta_gold & retrieved) / len(delta_gold) if delta_gold else None,
            })

        def total(key: str) -> float:
            return float(sum(r[key] for r in rows))

        def savings(ours_key: str, rest_key: str) -> Optional[float]:
            return 1.0 - total(ours_key) / total(rest_key) if total(rest_key) else None

        recalls = [r["delta_recall@5"] for r in rows if r["delta_recall@5"] is not None]
        return {
            "sessions": len(rows),
            "ours_retrieval_calls": total("ours_retrieval_calls"),
            "restart_retrieval_calls": total("restart_retrieval_calls"),
            "retrieval_call_ratio": total("ours_retrieval_calls") / total("restart_retrieval_calls") if total("restart_retrieval_calls") else None,
            "ours_tokens": total("ours_tokens"),
            "restart_tokens": total("restart_tokens"),
            "token_ratio": total("ours_tokens") / total("restart_tokens") if total("restart_tokens") else None,
            "refinement_savings_calls": savings("ours_retrieval_calls", "restart_retrieval_calls"),
            "refinement_savings_tokens": savings("ours_tokens", "restart_tokens"),
            "relation_accuracy": sum(1 for r in rows if r["relation"] == r["gold_relation"]) / len(rows),
            "affected_accuracy": sum(1 for r in rows if sorted(r["affected"] or []) == sorted(r["gold_affected"] or [])) / len(rows),
            "delta_fallbacks": sum(1 for r in rows if r["fallback"]),
            "delta_recall@5": sum(recalls) / len(recalls) if recalls else None,
            "restart_baseline": "b1 (restart_on_late_detail: re-retrieves the original and the combined request, re-drafts)",
            "per_session": rows,
        }

    def a3_experiment(self, out_dir: Optional[str] = None) -> Dict[str, Any]:
        """A3: the same scenarios under speculation.mode off | retrieval_only | full (all other settings
        equal, planner on in every arm). Latency is the replay clock's latency model (virtual time),
        not a hardware measurement."""
        scenarios = self.load()
        category = {tid: sc.get("category", "") for sc in scenarios for tid in turn_ids(sc)}
        arms: Dict[str, Any] = {}
        for arm in A3_ARMS:
            runner = ReplayRunner(
                self.scenarios_path, mode="ours", out_path=self.out_path, speed=self.speed, corpus_paths=self.corpus_paths,
                split=self.split, category=self.category, config_overrides={**self.config_overrides, "speculation.mode": arm},
            )
            events = runner._execute_scenarios("ours", scenarios)
            if out_dir:
                path = Path(out_dir) / f"a3_{arm}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    for d in events:
                        f.write(json.dumps(d, ensure_ascii=False) + "\n")
            m = MetricCalculator.calculate_all(events)
            row: Dict[str, Any] = {k: m.get(k) for k in A3_METRICS}
            for cat in ("compound", "single-early"):
                sub = MetricCalculator.calculate_all([e for e in events if category.get(e.get("turn_id", "")) == cat])
                row[f"ttft_p50_{cat}"], row[f"ttft_p95_{cat}"] = sub["ttft_p50"], sub["ttft_p95"]
            arms[arm] = row
        off_g, full_g = arms["off"].get("groundedness"), arms["full"].get("groundedness")
        return {
            "split": self.split, "category": self.category, "scenarios": len(scenarios), "arms": arms,
            "groundedness_parity": None if off_g is None or full_g is None else full_g >= off_g - 0.02,
            "timing": "virtual replay clock (LatencyModel), not wall-clock hardware latency",
        }

    def run_metrics(self) -> Dict[str, Any]:
        """Replay the loaded scenarios in this runner's mode and return the metrics (no files, no baselines)."""
        return MetricCalculator.calculate_all(self._execute_scenarios(self.mode, self.load()))

    def dry_run(self) -> Dict[str, Any]:
        """Load and validate inputs and build the engines without replaying any turn."""
        scenarios = self.load()
        index = self._index_for(self.mode)
        build_turn_engine(self.mode, index, self.bus, clock=self.clock, overrides=self.config_overrides)
        return {"mode": self.mode, "scenarios": len(scenarios), "indexed_chunks": len(index.chunks_map), **self._scenario_counts()}

    def run(self, generate_report: bool = True) -> Dict[str, Any]:
        p = Path(self.scenarios_path)
        if not p.exists() and p.suffix == ".jsonl":
            write_split_files(str(p.parent))
        scenarios = self.load()

        # 1. Target mode
        raw_events = self._execute_scenarios(self.mode, scenarios)
        out_p = Path(self.out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            for d in raw_events:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        metrics = MetricCalculator.calculate_all(raw_events)
        if not mode_config(self.mode).enable_verifier:
            metrics["groundedness"] = None

        # 2. In 'ours' mode, genuinely execute B1 and B0 on the same scenarios for the gates
        baseline_metrics: Dict[str, Any] = {}
        experiments: Dict[str, Any] = {}
        has_late = any(sc.get("follow_up") for sc in scenarios)
        if generate_report and self.mode == "ours":
            b1_events = self._execute_scenarios("b1", scenarios)
            b1_metrics = MetricCalculator.calculate_all(b1_events)
            if has_late:
                experiments["A4"] = self.a4_experiment(scenarios, raw_events, b1_events)
            b0_metrics = MetricCalculator.calculate_all(self._execute_scenarios("b0", scenarios))
            b0_metrics["groundedness"] = None
            baseline_metrics = {"b1": b1_metrics, "b0": b0_metrics}
            gates = GateEvaluator.evaluate_gates(metrics, b1_metrics, b0_metrics)
        else:
            gates = GateEvaluator.evaluate_gates(metrics, dict(metrics), dict(metrics))
            if self.mode == "ours" and has_late:
                experiments["A4"] = self.a4_experiment(scenarios, raw_events)

        if generate_report:
            cal_config = {
                "mode": self.mode,
                "seed": 13,
                "temperature": 0.0,
                "enable_cascade": MODE_FLAGS[self.mode]["enable_cascade"],
                "enable_multi_intent": MODE_FLAGS[self.mode]["enable_multi_intent"],
                "latency_model": DEFAULT_LATENCY.__dict__,
                "overrides": self.config_overrides,
            }
            counts = self._scenario_counts()
            cal_config.update({"split": self.split, "category": self.category, "replayed_scenarios": len(scenarios),
                               "cfg_hash": DEFAULT_CONFIG.cfg_hash, "config_frozen": DEFAULT_CONFIG.frozen})
            ReportGenerator.write_summary(str(out_p.parent / "summary.json"), self.mode, metrics, gates, counts, cal_config,
                                          baseline_metrics, experiments)
            ReportGenerator.write_markdown_report(str(out_p.parent / "report.md"), self.mode, metrics, gates, counts, cal_config,
                                                  baseline_metrics, experiments)
            if experiments.get("A4"):
                with open(out_p.parent / "a4.json", "w", encoding="utf-8") as f:
                    json.dump({"split": self.split, **experiments["A4"]}, f, indent=2)

        return {"metrics": metrics, "gates": gates, "experiments": experiments}
