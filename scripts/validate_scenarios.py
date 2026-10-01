#!/usr/bin/env python3
"""Validate replay scenarios: schema, unique ids, split hygiene against split.json, gold chunk ids.

Late-detail files (late_tune.jsonl / late_test.jsonl) are checked against split.json's
"late_detail" section, and their two-turn `follow_up` gold (relation, affected sub-intents,
delta gold chunk ids) is validated too.

Usage: python scripts/validate_scenarios.py [eval/scenarios] [--corpus data/a.json --corpus data/b.json]
Exits 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from slrag.config import DEFAULT_CONFIG  # noqa: E402
from slrag.corpus.chunker import SectionAwareChunker  # noqa: E402
from slrag.corpus.loader import CorpusLoader  # noqa: E402
from slrag.replay.baselines import DEFAULT_CORPORA  # noqa: E402

REQUIRED_FIELDS = {"scenario_id": str, "category": str, "query": str, "sub_intents": list, "is_unanswerable": bool}
RELATIONS = {"modifies", "adds", "unrelated"}
MIN_TOTAL = 75


def corpus_chunk_ids(paths: List[Path]) -> set:
    docs = [d for p in paths if p.exists() for d in CorpusLoader.load_file(p)]
    return {c.chunk_id for c in SectionAwareChunker(DEFAULT_CONFIG.chunker).chunk_documents(docs)}


def validate(scenario_dir: Path, corpus_paths: List[Path]) -> List[str]:
    errors: List[str] = []
    files = sorted(scenario_dir.glob("*.jsonl"))
    if not files:
        return [f"no *.jsonl scenario files in {scenario_dir}"]

    chunk_ids = corpus_chunk_ids(corpus_paths)
    by_file: Dict[str, List[str]] = {}
    seen: Dict[str, str] = {}
    categories: Dict[str, int] = {}

    for path in files:
        ids: List[str] = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            where = f"{path.name}:{line_no}"
            try:
                sc: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{where}: invalid JSON ({exc})")
                continue
            for field, typ in REQUIRED_FIELDS.items():
                if not isinstance(sc.get(field), typ):
                    errors.append(f"{where}: field '{field}' missing or not {typ.__name__}")
            sid = sc.get("scenario_id", f"<{where}>")
            if sid in seen:
                errors.append(f"{where}: duplicate scenario_id '{sid}' (also in {seen[sid]})")
            seen[sid] = path.name
            ids.append(sid)
            categories[sc.get("category", "?")] = categories.get(sc.get("category", "?"), 0) + 1

            for sub in sc.get("sub_intents") or []:
                gold = sub.get("gold_chunk_ids")
                if not isinstance(gold, list):
                    errors.append(f"{where}: sub-intent '{sub.get('id')}' has no gold_chunk_ids list")
                    continue
                if sc.get("is_unanswerable") and gold:
                    errors.append(f"{where}: unanswerable scenario has gold chunks {gold}")
                if not sc.get("is_unanswerable") and not gold:
                    errors.append(f"{where}: answerable sub-intent '{sub.get('id')}' has no gold chunks")
                for cid in gold:
                    if cid not in chunk_ids:
                        errors.append(f"{where}: gold chunk '{cid}' is not in the indexed corpus")

            follow_up = sc.get("follow_up")
            if sc.get("category") == "late_detail" and not isinstance(follow_up, dict):
                errors.append(f"{where}: late_detail scenario has no follow_up turn")
            if isinstance(follow_up, dict):
                sub_ids = {sub.get("id") for sub in sc.get("sub_intents") or []}
                if not isinstance(follow_up.get("text"), str) or not follow_up.get("text", "").strip():
                    errors.append(f"{where}: follow_up.text missing")
                if follow_up.get("relation") not in RELATIONS:
                    errors.append(f"{where}: follow_up.relation must be one of {sorted(RELATIONS)}")
                affected = follow_up.get("affected_sub_intents")
                if not isinstance(affected, list) or not set(affected) <= sub_ids:
                    errors.append(f"{where}: follow_up.affected_sub_intents must be a subset of {sorted(sub_ids)}")
                if follow_up.get("relation") == "modifies" and not affected:
                    errors.append(f"{where}: a 'modifies' follow_up needs affected_sub_intents")
                for cid in follow_up.get("delta_gold_chunk_ids") or []:
                    if cid not in chunk_ids:
                        errors.append(f"{where}: delta gold chunk '{cid}' is not in the indexed corpus")
        by_file[path.stem] = ids

    total = sum(len(v) for v in by_file.values())
    if total < MIN_TOTAL:
        errors.append(f"only {total} scenarios (need >= {MIN_TOTAL})")

    split_path = scenario_dir.parent / "split.json"
    if not split_path.exists():
        errors.append(f"missing split manifest {split_path}")
    else:
        split = json.loads(split_path.read_text(encoding="utf-8"))
        late = split.get("late_detail") or {}
        sections = [("", split), ("late_", late)] if late or any(k.startswith("late_") for k in by_file) else [("", split)]
        for prefix, section in sections:
            tune, test = set(section.get("tune", [])), set(section.get("test", []))
            if tune & test:
                errors.append(f"split.json: {len(tune & test)} {prefix}scenario(s) in both tune and test: {sorted(tune & test)[:5]}")
            for name, expected in ((f"{prefix}tune", tune), (f"{prefix}test", test)):
                actual = set(by_file.get(name, []))
                if actual != expected:
                    errors.append(f"{name}.jsonl does not match split.json ({len(actual ^ expected)} id(s) differ)")

    print(f"scenario files: {[p.name for p in files]}")
    print(f"scenarios: {total} {dict(sorted(categories.items()))}")
    print(f"split: " + ", ".join(f"{k}={len(v)}" for k, v in by_file.items()))
    print(f"corpus chunks: {len(chunk_ids)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenario_dir", nargs="?", default=str(ROOT / "eval" / "scenarios"))
    parser.add_argument("--corpus", action="append", type=Path, help="corpus file(s); defaults to the replay corpora")
    args = parser.parse_args()

    errors = validate(Path(args.scenario_dir), args.corpus or list(DEFAULT_CORPORA))
    for err in errors:
        print(f"ERROR {err}")
    print("OK" if not errors else f"FAILED: {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
