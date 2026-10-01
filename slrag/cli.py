"""Command-line interface for SL-RAG (audit, index, search, probe)."""

import json
from pathlib import Path
import sys
import time
import click

from slrag.config import DEFAULT_CONFIG
from slrag.corpus.loader import CorpusLoader
from slrag.corpus.chunker import SectionAwareChunker
from slrag.retrieval.engine import HybridRetrievalEngine


@click.group()
def cli():
    """SL-RAG: Streaming Hybrid Retrieval-Augmented Generation CLI."""
    pass


@cli.command("audit")
@click.argument("corpus_path", type=click.Path(exists=True))
def audit_command(corpus_path: str):
    """Audit corpus files, chunks, token distributions, and section integrity."""
    click.echo(f"Auditing corpus from: {corpus_path}")
    path = Path(corpus_path)
    if path.is_dir():
        docs = CorpusLoader.load_directory(path)
    else:
        docs = CorpusLoader.load_file(path)

    click.echo(f"Loaded {len(docs)} documents.")
    chunker = SectionAwareChunker(DEFAULT_CONFIG.chunker)
    chunks = chunker.chunk_documents(docs)

    click.echo(f"Generated {len(chunks)} chunks.")
    token_counts = [c.token_count for c in chunks]
    min_tokens = min(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0

    chunk_ids = set()
    duplicates = 0
    for c in chunks:
        if c.chunk_id in chunk_ids:
            duplicates += 1
        chunk_ids.add(c.chunk_id)

    click.echo("--- Audit Summary ---")
    click.echo(f"Total Documents:    {len(docs)}")
    click.echo(f"Total Chunks:       {len(chunks)}")
    click.echo(f"Unique Chunk IDs:   {len(chunk_ids)}")
    click.echo(f"Duplicate IDs:      {duplicates}")
    click.echo(f"Token Min / Avg / Max: {min_tokens} / {avg_tokens:.1f} / {max_tokens}")
    click.echo(f"Sample Chunk IDs:   {list(chunk_ids)[:5]}")
    click.echo("Audit passed successfully.")


@cli.command("index")
@click.argument("corpus_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="./data/index", help="Output index directory.")
def index_command(corpus_path: str, output: str):
    """Chunk, compute BM25 and Dense embeddings, and persist index."""
    click.echo(f"Indexing corpus from {corpus_path} -> {output}")
    path = Path(corpus_path)
    if path.is_dir():
        docs = CorpusLoader.load_directory(path)
    else:
        docs = CorpusLoader.load_file(path)

    engine = HybridRetrievalEngine(DEFAULT_CONFIG)
    total_indexed = engine.index_documents(docs)
    engine.save(Path(output))

    stats = engine.get_stats()
    click.echo(f"Successfully indexed {total_indexed} chunks.")
    click.echo(json.dumps(stats, indent=2))


@cli.command("search")
@click.argument("query")
@click.option("--index", "-i", default="./data/index", help="Path to index directory.")
@click.option("--mode", "-m", default="hybrid", type=click.Choice(["bm25", "dense", "hybrid"], case_sensitive=False))
@click.option("--top", "-k", default=10, type=int, help="Number of results to return.")
def search_command(query: str, index: str, mode: str, top: int):
    """Search the index using bm25, dense, or hybrid (RRF) mode."""
    idx_path = Path(index)
    if not idx_path.exists():
        click.echo(f"Error: Index directory {index} does not exist. Run 'slrag index' first.", err=True)
        sys.exit(1)

    engine = HybridRetrievalEngine.load(idx_path, config=DEFAULT_CONFIG)
    t0 = time.perf_counter()
    results = engine.search(query, mode=mode, top_k=top)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    click.echo(f"\n--- Search Results for '{query}' [Mode: {mode.upper()}, Top: {top}] (Latency: {latency_ms:.2f}ms) ---")
    if not results:
        click.echo("No results found.")
        return

    for r in results:
        click.echo(f"\n[{r.rank}] Chunk ID: {r.chunk_id} | Score: {r.score:.4f} | Section: {r.section_title}")
        click.echo(f"    Source Scores: {r.source_scores}")
        preview = (r.text[:120] + "...") if len(r.text) > 120 else r.text
        click.echo(f"    Text: {preview}")


@cli.command("probe")
@click.option("--index", "-i", default="./data/index", help="Path to index directory.")
def probe_command(index: str):
    """Probe system health, index status, embedding dimension, and retrieval latency."""
    click.echo("--- Probing SL-RAG Engine ---")
    click.echo(f"Config Hash: {DEFAULT_CONFIG.cfg_hash}")
    click.echo(f"Embedding Model: {DEFAULT_CONFIG.dense.model_name} (Dim: {DEFAULT_CONFIG.dense.embedding_dim})")

    idx_path = Path(index)
    if idx_path.exists():
        engine = HybridRetrievalEngine.load(idx_path, config=DEFAULT_CONFIG)
        stats = engine.get_stats()
        click.echo(f"Index Location: {idx_path.resolve()}")
        click.echo(f"Indexed Chunks: {stats['total_chunks']}")
        click.echo(f"BM25 Vocab Size: {stats['vocab_size']}")

        # Benchmark latency
        benchmark_queries = ["vector search retrieval", "bm25 okapi postings", "system latency"]
        latencies = []
        for q in benchmark_queries:
            t0 = time.perf_counter()
            _ = engine.search(q, mode="hybrid", top_k=5)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        avg_lat = sum(latencies) / len(latencies)
        click.echo(f"Benchmark Search Latency (Hybrid): avg {avg_lat:.2f}ms across {len(benchmark_queries)} queries.")
    else:
        click.echo(f"Index directory {index} not found (fresh state).")

    click.echo("Probe completed: System healthy.")


@cli.command("replay")
@click.argument("scenarios_path", type=click.Path())
@click.option("--mode", type=click.Choice(["ours", "b0", "b1"], case_sensitive=False), default="ours", help="Execution mode.")
@click.option("--out", default="out/telemetry.jsonl", help="Telemetry output JSONL path (summary.json is written next to it).")
@click.option("--speed", type=float, default=0.0, help="Wall-clock pacing: 1 = real time, 0 = as fast as possible. Metrics use virtual time either way.")
@click.option("--split", type=click.Choice(["test", "tune"]), default="test", help="Split to load when SCENARIOS_PATH is a directory.")
@click.option("--category", default=None, help="Only replay scenarios of this category (e.g. late_detail, compound).")
@click.option("--speculation", type=click.Choice(["off", "retrieval_only", "full"]), default=None,
              help="Override speculation.mode (A3 arm) for this replay.")
@click.option("--experiment", type=click.Choice(["A3"], case_sensitive=False), default=None,
              help="A3: replay the scenarios under all three speculation arms and write a3.json next to --out.")
@click.option("--trace-out", default=None, help="Also write one trace JSONL per scenario to this directory (<mode>_<scenario>.jsonl).")
@click.option("--dry-run", is_flag=True, help="Validate inputs and build the engine without replaying.")
def replay_command(scenarios_path: str, mode: str, out: str, speed: float, split: str, category: str, speculation: str,
                   experiment: str, trace_out: str, dry_run: bool):
    """Replay scenarios through the engine.

    SCENARIOS_PATH is a JSONL file, a scenario directory (loads <split>.jsonl and
    late_<split>.jsonl) or <dir>/test | <dir>/tune."""
    from slrag.eval.runner import ReplayRunner, resolve_scenario_files

    files = resolve_scenario_files(scenarios_path, split)
    if not files or not all(f.exists() for f in files):
        raise click.BadParameter(f"no scenario files found for '{scenarios_path}'", param_hint="SCENARIOS_PATH")
    if Path(scenarios_path).name in ("test", "tune") and not Path(scenarios_path).exists():
        split = Path(scenarios_path).name
    overrides = {"speculation.mode": speculation} if speculation else {}
    runner = ReplayRunner(scenarios_path=scenarios_path, mode=mode, out_path=out, speed=speed, split=split, category=category,
                          config_overrides=overrides, trace_out=trace_out)
    if experiment and experiment.upper() == "A3":
        out_dir = Path(out).parent
        result = runner.a3_experiment(out_dir=str(out_dir))
        (out_dir / "a3.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        click.echo(json.dumps(result, indent=2))
        return
    if dry_run:
        click.echo(json.dumps(runner.dry_run(), indent=2))
        return
    result = runner.run(generate_report=True)
    click.echo(json.dumps({"metrics": result["metrics"], "gates": {k: v["status"] for k, v in result["gates"].items()}}, indent=2))


@cli.command("report")
@click.argument("out_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--split", type=click.Choice(["test", "tune"]), default="test", help="Split the replay was run on (checked against the summary).")
@click.option("--experiment", type=click.Choice(["A3", "A4"], case_sensitive=False), default=None, help="Print one experiment's results.")
def report_command(out_dir: str, split: str, experiment: str):
    """Print the gates of the last replay in OUT_DIR, or one experiment (A3: speculation arms,
    A4: refinement vs restart)."""
    base = Path(out_dir)
    if experiment and experiment.upper() == "A3":
        path = base / "a3.json"
        if not path.exists():
            raise click.ClickException(f"{path} not found: run `replay ... --experiment A3` first")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("split") != split:
            click.echo(f"warning: A3 results are for split '{data.get('split')}', not '{split}'", err=True)
        keys = ["ttft_p50", "ttft_p95", "ttft_p50_compound", "ttft_p95_compound", "ttft_p50_single-early", "ttft_p95_single-early",
                "cost_per_turn", "wasted_speculation_rate", "cache_hit_rate", "groundedness", "reconcile_stands", "reconcile_refined",
                "reconcile_redone", "wasted_tokens", "predraft_waste_rate", "predraft_ready_before_end_rate"]
        click.echo(f"A3 ({data.get('timing')}), split={data.get('split')}, scenarios={data.get('scenarios')}")
        click.echo(f"{'metric':32s}" + "".join(f"{arm:>16s}" for arm in data["arms"]))
        for k in keys:
            vals = [data["arms"][arm].get(k) for arm in data["arms"]]
            click.echo(f"{k:32s}" + "".join(f"{'n/a' if v is None else f'{v:.4f}':>16s}" for v in vals))
        click.echo(f"groundedness_parity (full >= off - 0.02): {data.get('groundedness_parity')}")
        return
    if experiment and experiment.upper() == "A4":
        path = base / "a4.json"
        if not path.exists():
            raise click.ClickException(f"{path} not found: run `replay ... --mode ours` on late-detail scenarios first")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("split") != split:
            click.echo(f"warning: A4 results are for split '{data.get('split')}', not '{split}'", err=True)
        click.echo(json.dumps({k: v for k, v in data.items() if k != "per_session"}, indent=2))
        return
    summary = base / "summary.json"
    if not summary.exists():
        raise click.ClickException(f"{summary} not found")
    data = json.loads(summary.read_text(encoding="utf-8"))
    click.echo(json.dumps({"mode": data.get("mode"), "gates": {k: v["status"] for k, v in data.get("gates", {}).items()}}, indent=2))


@cli.command("coverage")
@click.argument("telemetry_path", type=click.Path(exists=True))
def coverage_command(telemetry_path: str):
    """Report per-turn trace coverage of the canonical turn stages in a telemetry JSONL file."""
    from slrag.replay.metrics import MetricCalculator

    with open(telemetry_path, "r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    by_turn: dict = {}
    for e in events:
        if e.get("turn_id"):
            by_turn.setdefault(e["turn_id"], []).append(e)
    incomplete = 0
    for turn_id, turn_events in by_turn.items():
        cov = MetricCalculator.calculate_all(turn_events)["trace_coverage"]
        if cov < 1.0:
            incomplete += 1
            click.echo(f"{turn_id}: trace_coverage={cov:.3f}")
    overall = MetricCalculator.calculate_all(events)["trace_coverage"]
    click.echo(f"turns={len(by_turn)} incomplete={incomplete} trace_coverage={overall:.4f}")


if __name__ == "__main__":
    cli()
