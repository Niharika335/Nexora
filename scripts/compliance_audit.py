#!/usr/bin/env python3
"""Compliance audit for the hard rules (Phase 10). Writes out/compliance.json and exits 1 on any FAIL.

Six checks, each PASS/FAIL with the offending file:line for every finding:
  1. corpus_isolation    the engine makes no network calls except the configured LLM endpoint (model
                         files load with local_files_only=True); no
                         web-search packages in the dependencies; every cite emitted in the test run
                         exists in the corpus index.
  2. no_hardcoding       no example strings from the brief, no corpus text (>= 8 consecutive words),
                         no scenario queries and no corpus chunk ids in engine code; eval/ and tests/
                         are kept out of the Docker image.
  3. no_precomputation   engine modules never read eval/ scenario files; replaying the test split with every
                         gold label stripped gives identical answers and retrievals (labels only annotate
                         telemetry for the metrics); run outputs (out/) are not tracked by git;
                         summary.json was produced by a run with the current cfg_hash.
  4. session_isolation   two interleaved live sessions share no ledger, evidence cache, reorder buffer or
                         normalizer; each ledger only holds its own turns; sessions are gone after
                         session_end; no per-user files are written.
  5. parsimony           no agent-framework or orchestration libraries in the dependencies or imports;
                         LLM calls per turn by turn type, from the test-run telemetry.
  6. prompts             LLM prompt templates and few-shot exemplars contain no example strings from the
                         brief, no corpus text and no scenario text; exemplar ids are placeholders.

Usage: python scripts/compliance_audit.py [--telemetry out/ours_test.jsonl] [--out out/compliance.json]
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PKG = ROOT / "slrag"
# Offline harness / UI modules may read scenarios; the engine (everything that serves a live turn) may not.
HARNESS_DIRS = {"eval", "replay", "server", "ui"}
HARNESS_FILES = {"cli.py"}
NETWORK_MODULES = {"httpx", "requests", "aiohttp", "urllib.request", "http.client", "socket", "websocket", "websockets"}
NETWORK_ALLOWED = {"slrag/llm/ollama_client.py"}  # talks only to llm.ollama_url
HUB_MODULES = {"sentence_transformers", "transformers", "huggingface_hub"}  # must load from the local model cache
WEB_SEARCH_PACKAGES = {"duckduckgo-search", "duckduckgo_search", "googlesearch-python", "google-search-results", "serpapi",
                       "tavily-python", "tavily", "exa-py", "wikipedia", "bing-search", "brave-search", "searx", "newspaper3k"}
AGENT_FRAMEWORKS = {"langchain", "langchain-core", "langchain-community", "langgraph", "llama-index", "llama_index", "autogen",
                    "pyautogen", "crewai", "haystack-ai", "farm-haystack", "semantic-kernel", "dspy", "dspy-ai", "guidance",
                    "smolagents", "agno", "phidata", "camel-ai", "openai-agents", "pydantic-ai", "instructor", "marvin"}
BRIEF_EXAMPLES = ["Pune", "Venue A", "catering", "reimbursement", "Doc_12", "Doc_31", "Doc_09"]
NGRAM = 8
WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def code_files() -> List[Path]:
    return sorted(p for p in PKG.rglob("*") if p.suffix in (".py", ".html", ".json") and "__pycache__" not in p.parts)


def engine_file(path: Path) -> bool:
    parts = path.resolve().relative_to(PKG).parts
    return parts[0] not in HARNESS_DIRS and parts[0] not in HARNESS_FILES


def words(text: str) -> List[str]:
    return WORD.findall(text.lower())


def ngrams(tokens: List[str], n: int) -> Iterable[Tuple[str, ...]]:
    return (tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def load_corpus() -> List[Dict[str, Any]]:
    from slrag.replay.baselines import DEFAULT_CORPORA

    docs = []
    for path in DEFAULT_CORPORA:
        if Path(path).exists():
            docs.extend(json.loads(Path(path).read_text(encoding="utf-8")))
    return docs


def load_scenarios() -> List[Dict[str, Any]]:
    out = []
    for f in sorted((ROOT / "eval" / "scenarios").glob("*.jsonl")):
        out.extend(json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip())
    return out


def string_literals(path: Path) -> List[Tuple[int, str]]:
    """(line, text) of every string constant in a Python file; whole-file lines for other files."""
    text = path.read_text(encoding="utf-8")
    if path.suffix != ".py":
        return list(enumerate(text.splitlines(), 1))
    out = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append((node.lineno, node.value))
    return out


def dependency_names() -> Dict[str, str]:
    """Dependency name -> where it is declared (pyproject.toml, requirements.lock)."""
    import tomllib

    names: Dict[str, str] = {}
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    for spec in project.get("dependencies", []) + [d for ds in project.get("optional-dependencies", {}).values() for d in ds]:
        names[re.split(r"[<>=!~\[; ]", spec, maxsplit=1)[0].lower()] = "pyproject.toml"
    lock = ROOT / "requirements.lock"
    if lock.exists():
        for i, line in enumerate(lock.read_text(encoding="utf-8").splitlines(), 1):
            m = re.match(r"^([A-Za-z0-9_.\-]+)==", line)
            if m:
                names[m.group(1).lower()] = f"requirements.lock:{i}"
    return names


def imports(path: Path) -> List[Tuple[int, str]]:
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out.extend((node.lineno, a.name) for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append((node.lineno, node.module))
    return out


def result(findings: List[str], details: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "FAIL" if findings else "PASS", "findings": findings, "details": details}


# 1 ------------------------------------------------------------------------------------------------
def check_corpus_isolation(telemetry: Path) -> Dict[str, Any]:
    from slrag.replay.baselines import build_index, mode_config

    findings: List[str] = []
    for path in code_files():
        if path.suffix != ".py" or rel(path) in NETWORK_ALLOWED:
            continue
        for line, mod in imports(path):
            if mod in NETWORK_MODULES or mod.split(".")[0] in {"httpx", "requests", "aiohttp", "websocket"}:
                findings.append(f"{rel(path)}:{line} imports network client '{mod}'")
    for path in code_files():
        if path.suffix != ".py":
            continue
        hub = [(line, mod) for line, mod in imports(path) if mod.split(".")[0] in HUB_MODULES]
        if hub and "local_files_only=True" not in path.read_text(encoding="utf-8"):
            findings.append(f"{rel(path)}:{hub[0][0]} loads '{hub[0][1]}' without local_files_only=True (could download at run time)")
    for name, where in dependency_names().items():
        if name in WEB_SEARCH_PACKAGES:
            findings.append(f"{where} declares web-search package '{name}'")

    cites: Set[str] = set()
    if telemetry.exists():
        for line in telemetry.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if e.get("event_type") == "answer_chunk":
                cites.update(c for c in e.get("cites") or [] if c)
    else:
        findings.append(f"{rel(telemetry)} missing: run the test split first")
    known = set(build_index(mode_config("ours").app).chunks_map)
    findings.extend(f"{rel(telemetry)} cites '{c}', which is not in the corpus index" for c in sorted(cites - known))
    return result(findings, {"network_client_allowed_in": sorted(NETWORK_ALLOWED), "emitted_cites": len(cites),
                             "cites_in_index": len(cites & known), "indexed_chunks": len(known)})


# 2 ------------------------------------------------------------------------------------------------
def check_no_hardcoding(corpus_grams: Set[Tuple[str, ...]], scenarios: List[Dict[str, Any]], chunk_ids: Set[str]) -> Dict[str, Any]:
    findings: List[str] = []
    examples = [(s, re.compile(rf"\b{re.escape(s)}\b", re.IGNORECASE)) for s in BRIEF_EXAMPLES]
    queries = {" ".join(words(q)) for sc in scenarios for q in [sc.get("query", "")] + [s.get("query", "") for s in sc.get("sub_intents", [])]
               + [(sc.get("follow_up") or {}).get("text", "")] if len(words(q)) >= 4}
    for path in code_files():
        for line, text in string_literals(path):
            for name, pat in examples:
                if pat.search(text):
                    findings.append(f"{rel(path)}:{line} contains brief example string '{name}'")
            if not engine_file(path):
                continue  # harness / UI code may name scenarios and chunks it displays
            toks = words(text)
            hit = next((g for g in ngrams(toks, NGRAM) if g in corpus_grams), None)
            if hit:
                findings.append(f"{rel(path)}:{line} contains corpus text: '{' '.join(hit)}'")
            joined = " ".join(toks)
            for q in queries:
                if q and q in joined:
                    findings.append(f"{rel(path)}:{line} contains scenario query text: '{q[:60]}'")
            for cid in chunk_ids:
                if cid in text:
                    findings.append(f"{rel(path)}:{line} hardcodes corpus chunk id '{cid}'")

    dockerignore = ROOT / ".dockerignore"
    excluded = {l.strip().rstrip("/") for l in dockerignore.read_text(encoding="utf-8").splitlines()} if dockerignore.exists() else set()
    for required in ("eval", "tests", "out", "docs"):
        if required not in excluded:
            findings.append(f".dockerignore does not exclude {required}/")
    dockerfile = ROOT / "Dockerfile"
    if dockerfile.exists():
        for i, line in enumerate(dockerfile.read_text(encoding="utf-8").splitlines(), 1):
            if re.match(r"\s*(COPY|ADD)\b", line, re.IGNORECASE) and re.search(r"\b(eval|tests)\b", line):
                findings.append(f"Dockerfile:{i} copies evaluation data into the image")
    else:
        findings.append("Dockerfile missing")
    return result(findings, {"brief_examples": BRIEF_EXAMPLES, "corpus_ngram": NGRAM, "scenario_queries_checked": len(queries),
                             "chunk_ids_checked": len(chunk_ids), "dockerignore": sorted(excluded),
                             "image_listing": "static check of Dockerfile + .dockerignore (docker not required)"})


GOLD_KEYS = ("gold_chunk_ids", "is_unanswerable", "delta_gold_chunk_ids", "expected_chunk_ids")


def _strip_gold(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_gold(v) for k, v in value.items() if k not in GOLD_KEYS}
    if isinstance(value, list):
        return [_strip_gold(v) for v in value]
    return value


def _decisions(events: List[Dict[str, Any]]) -> List[Tuple[Any, ...]]:
    """What the engine decided: retrieval queries and results, and every emitted claim with its cites."""
    out = []
    for e in events:
        t = e.get("event_type")
        if t == "answer_chunk":
            out.append((t, e.get("turn_id"), e.get("text"), tuple(e.get("cites") or [])))
        elif t in ("speculative_retrieval", "sub_intent_retrieval"):
            out.append((t, e.get("turn_id"), e.get("retrieval_query") or e.get("query"), tuple(e.get("retrieved_chunk_ids") or [])))
        elif t == "turn_complete":
            out.append((t, e.get("turn_id"), e.get("retrieval_calls"), e.get("llm_calls")))
    return out


def gold_labels_do_not_change_outputs() -> Dict[str, Any]:
    """Replay the test split twice, with and without the gold labels; the decisions must match."""
    from slrag.eval.runner import ReplayRunner

    runner = ReplayRunner(str(ROOT / "eval" / "scenarios"), mode="ours", split="test")
    scenarios = runner.load()
    with_labels = _decisions(runner._execute_scenarios("ours", scenarios))
    without = _decisions(runner._execute_scenarios("ours", [_strip_gold(sc) for sc in scenarios]))
    findings = []
    if with_labels != without:
        first = next((i for i, (a, b) in enumerate(zip(with_labels, without)) if a != b), min(len(with_labels), len(without)))
        findings.append(f"gold labels change engine output (first difference at decision {first}: "
                        f"{with_labels[first] if first < len(with_labels) else None} vs {without[first] if first < len(without) else None})")
    return {"findings": findings, "scenarios": len(scenarios), "decisions_compared": len(with_labels), "identical": not findings}


# 3 ------------------------------------------------------------------------------------------------
def check_no_precomputation(summary: Path) -> Dict[str, Any]:
    from slrag.config import DEFAULT_CONFIG

    findings: List[str] = []
    pat = re.compile(r"eval[/\\]scenarios|scenarios[/\\]|split\.json|\b(late_)?(tune|test)\.jsonl\b|gold_chunk_ids")
    for path in code_files():
        if path.suffix != ".py" or not engine_file(path):
            continue
        for line, text in string_literals(path):
            if pat.search(text):
                findings.append(f"{rel(path)}:{line} engine module references evaluation data: {text.strip()[:60]!r}")
    label_check = gold_labels_do_not_change_outputs()
    findings.extend(label_check.pop("findings"))
    tracked = subprocess.run(["git", "ls-files", "out", "summary.json", "results.jsonl", "telemetry.jsonl"],
                             cwd=ROOT, capture_output=True, text=True).stdout.split()
    findings.extend(f"{t} is tracked by git (run outputs must be produced by the run)" for t in tracked)
    ignored = subprocess.run(["git", "check-ignore", "-q", "out/summary.json"], cwd=ROOT).returncode == 0
    if not ignored:
        findings.append(".gitignore does not ignore out/ (summary.json could be shipped)")
    details: Dict[str, Any] = {"out_ignored_by_git": ignored, "gold_label_strip_replay": label_check}
    if summary.exists():
        data = json.loads(summary.read_text(encoding="utf-8"))
        produced = (data.get("calibration_config") or {}).get("cfg_hash")
        details.update({"summary_cfg_hash": produced, "current_cfg_hash": DEFAULT_CONFIG.cfg_hash})
        if produced != DEFAULT_CONFIG.cfg_hash:
            findings.append(f"{rel(summary)} cfg_hash {produced} != current {DEFAULT_CONFIG.cfg_hash}: re-run the replay")
    else:
        findings.append(f"{rel(summary)} missing: run the test split first")
    return result(findings, details)


# 4 ------------------------------------------------------------------------------------------------
def check_session_isolation() -> Dict[str, Any]:
    from starlette.testclient import TestClient

    from slrag.config import DEFAULT_CONFIG, TelemetryConfig
    from slrag.gateway.session import SessionRegistry
    from slrag.pipeline.turn_engine import BatchTurnEngine
    from slrag.replay.baselines import build_index, mode_config
    from slrag.server.app import create_app
    from slrag.telemetry.bus import TelemetryBus
    from slrag.telemetry.metrics import MetricsCollector

    findings: List[str] = []
    turns = {"audit_a": ["How long is attendee personal data", "retained after an event?"],
             "audit_b": ["What are the API rate limits", "for Growth workspaces?"]}
    with tempfile.TemporaryDirectory() as tmp:
        cwd_before = set(os.listdir(ROOT))
        index = build_index(mode_config("ours").app)
        bus = TelemetryBus(TelemetryConfig(jsonl_log_path=str(Path(tmp) / "audit.jsonl")))
        metrics = MetricsCollector()
        registry = SessionRegistry()
        app = create_app(config=DEFAULT_CONFIG, engine=index, bus=bus, registry=registry, metrics=metrics,
                         turn_engine=BatchTurnEngine(DEFAULT_CONFIG, engine=index, bus=bus, metrics=metrics))
        state: Dict[str, Dict[str, Any]] = {}
        with TestClient(app) as client:
            with client.websocket_connect("/ws/stream?session_id=audit_a") as wa, \
                    client.websocket_connect("/ws/stream?session_id=audit_b") as wb:
                sockets = {"audit_a": wa, "audit_b": wb}
                for i in range(2):  # interleave the two sessions chunk by chunk
                    for sid, ws in sockets.items():
                        ws.send_text(json.dumps({"event_type": "transcript_chunk", "text": turns[sid][i], "seq": i + 1}))
                for sid, ws in sockets.items():
                    ws.send_text(json.dumps({"event_type": "utterance_end", "seq": 3}))
                    while json.loads(ws.receive_text())["event_type"] != "turn_summary":
                        pass
                for sid in sockets:
                    ctx = registry._sessions[sid]
                    ledger = ctx.custom_state.get("ledger")
                    state[sid] = {"ctx": ctx, "ledger": ledger, "cache": ctx.custom_state.get("evidence_cache"),
                                  "turns": {getattr(c, "turn_id", "") for c in (ledger.get_verified_claims() if ledger else [])}}
                a, b = state["audit_a"], state["audit_b"]
                for key, label in (("ctx", "session context"), ("ledger", "claim ledger"), ("cache", "evidence cache")):
                    if a[key] is not None and a[key] is b[key]:
                        findings.append(f"slrag/gateway/session.py: both sessions share one {label}")
                for attr in ("reorder_buffer", "normalizer", "custom_state"):
                    if getattr(a["ctx"], attr) is getattr(b["ctx"], attr):
                        findings.append(f"slrag/gateway/session.py: both sessions share {attr}")
                for sid, other in (("audit_a", "audit_b"), ("audit_b", "audit_a")):
                    leaked = [t for t in state[sid]["turns"] if t and other in t]
                    if leaked:
                        findings.append(f"session {sid} ledger holds claims of {other}: {leaked}")
                for ws in sockets.values():
                    ws.send_text(json.dumps({"event_type": "session_end", "seq": 4}))
        if registry.active_sessions_count:
            findings.append(f"{registry.active_sessions_count} session(s) still registered after session_end")
        new_files = sorted(set(os.listdir(ROOT)) - cwd_before)
        findings.extend(f"per-user file written to the repo root: {f}" for f in new_files)
        details = {"sessions": sorted(turns), "ledger_claims": {sid: len(s["turns"]) for sid, s in state.items()},
                   "active_after_session_end": registry.active_sessions_count,
                   "idle_ttl_s": DEFAULT_CONFIG.session.idle_timeout_seconds}
    return result(findings, details)


# 5 ------------------------------------------------------------------------------------------------
def check_parsimony(telemetry: Path, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    findings: List[str] = []
    for name, where in dependency_names().items():
        if name in AGENT_FRAMEWORKS:
            findings.append(f"{where} declares agent/orchestration framework '{name}'")
    roots = {n.replace("-", "_") for n in AGENT_FRAMEWORKS}
    for path in code_files():
        if path.suffix == ".py":
            for line, mod in imports(path):
                if mod.split(".")[0] in roots:
                    findings.append(f"{rel(path)}:{line} imports agent framework '{mod}'")

    category = {sc["scenario_id"]: sc.get("category", "") for sc in scenarios}
    calls: Dict[str, List[int]] = defaultdict(list)
    if telemetry.exists():
        for line in telemetry.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if e.get("event_type") == "turn_complete":
                tid = e.get("turn_id", "")
                base, _, part = tid.partition(":")
                kind = category.get(base, "unknown") + (" (refinement turn)" if part == "t2" else "")
                calls[kind].append(int(e.get("llm_calls") or 0))
    per_type = {k: {"turns": len(v), "mean_llm_calls": round(sum(v) / len(v), 3), "max_llm_calls": max(v)}
                for k, v in sorted(calls.items())}
    return result(findings, {"dependencies": sorted(dependency_names()), "llm_calls_per_turn_by_type": per_type})


# 6 ------------------------------------------------------------------------------------------------
def prompt_literals() -> List[Tuple[Path, int, str]]:
    """Every string literal that is (part of) an LLM prompt: module constants named *PROMPT*, strings
    inside functions whose name mentions prompt, and system / message content literals."""
    out = []
    for path in code_files():
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and "PROMPT" in t.id for t in node.targets):
                out.extend((path, n.lineno, n.value) for n in ast.walk(node.value) if isinstance(n, ast.Constant) and isinstance(n.value, str))
            elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "system_prompt" for t in node.targets):
                out.extend((path, n.lineno, n.value) for n in ast.walk(node.value) if isinstance(n, ast.Constant) and isinstance(n.value, str))
            elif isinstance(node, ast.Dict):
                keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
                if "role" in keys and "content" in keys:
                    out.extend((path, n.lineno, n.value) for n in ast.walk(node) if isinstance(n, ast.Constant)
                               and isinstance(n.value, str) and n.value not in ("role", "content", "system", "user", "assistant"))
    return out


def check_prompts(corpus_grams: Set[Tuple[str, ...]], scenarios: List[Dict[str, Any]], chunk_ids: Set[str]) -> Dict[str, Any]:
    findings: List[str] = []
    literals = prompt_literals()
    examples = [(s, re.compile(rf"\b{re.escape(s)}\b", re.IGNORECASE)) for s in BRIEF_EXAMPLES]
    scenario_grams = {g for sc in scenarios for g in ngrams(words(sc.get("query", "")), 5)}
    exemplar_ids: List[str] = []
    for path, line, text in literals:
        where = f"{rel(path)}:{line}"
        for name, pat in examples:
            if pat.search(text):
                findings.append(f"{where} prompt contains brief example string '{name}'")
        toks = words(text)
        if any(g in corpus_grams for g in ngrams(toks, NGRAM)):
            findings.append(f"{where} prompt contains corpus text")
        if any(g in scenario_grams for g in ngrams(toks, 5)):
            findings.append(f"{where} prompt contains scenario query text")
        for cid in re.findall(r"\[([^\]\s]+§[^\]\s]+)\]", text):
            exemplar_ids.append(cid)
            if cid in chunk_ids:
                findings.append(f"{where} few-shot exemplar cites real corpus chunk '{cid}'")
    return result(findings, {"prompt_literals": len(literals), "files": sorted({rel(p) for p, _, _ in literals}),
                             "exemplar_ids": sorted(set(exemplar_ids))})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--telemetry", default=str(ROOT / "out" / "ours_test.jsonl"), help="Test-split replay telemetry")
    parser.add_argument("--summary", default=str(ROOT / "out" / "summary.json"))
    parser.add_argument("--out", default=str(ROOT / "out" / "compliance.json"))
    args = parser.parse_args()

    docs = load_corpus()
    corpus_grams = {g for d in docs for s in d.get("sections", []) for g in ngrams(words(s.get("text", "")), NGRAM)}
    chunk_ids = {f"{d['doc_id']}§{s['section_id']}" for d in docs for s in d.get("sections", [])}
    scenarios = load_scenarios()
    telemetry, summary = Path(args.telemetry), Path(args.summary)

    checks = {
        "corpus_isolation": check_corpus_isolation(telemetry),
        "no_hardcoding": check_no_hardcoding(corpus_grams, scenarios, chunk_ids),
        "no_precomputation": check_no_precomputation(summary),
        "session_isolation": check_session_isolation(),
        "parsimony": check_parsimony(telemetry, scenarios),
        "prompts": check_prompts(corpus_grams, scenarios, chunk_ids),
    }
    passed = all(c["status"] == "PASS" for c in checks.values())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"status": "PASS" if passed else "FAIL", "checks": checks}, indent=2), encoding="utf-8")
    for name, c in checks.items():
        print(f"{c['status']:4}  {name}")
        for f in c["findings"]:
            print(f"      - {f}")
    print(f"wrote {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
