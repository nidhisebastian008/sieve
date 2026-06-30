from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="sieve — production traces → fine-tuning data pipeline", no_args_is_help=True)
dataset_app = typer.Typer(no_args_is_help=True)
app.add_typer(dataset_app, name="dataset", help="Dataset version management")

console = Console()


def _session():
    from sieve.db import init_db, get_session
    init_db()
    return get_session()


# ─── INGEST ────────────────────────────────────────────────────────────────────

@app.command()
def ingest(
    path: Path = typer.Argument(..., help="JSONL file to ingest"),
):
    """Ingest interactions from a JSONL file."""
    from sieve.ingest.jsonl import JSONLIngester

    session = _session()
    count = 0
    for interaction in JSONLIngester(path).ingest():
        session.add(interaction)
        count += 1
    session.commit()
    console.print(f"[green]✓[/green] Ingested [bold]{count}[/bold] interactions from {path}")


# ─── SCORE ─────────────────────────────────────────────────────────────────────

@app.command()
def score(
    scorer: str = typer.Option(
        "heuristic",
        help="Scorer: heuristic | llm-judge | ifd | combined",
    ),
    model: str = typer.Option("llama3.2", help="Ollama model for llm-judge / ifd scorers"),
    ollama_host: str = typer.Option("http://localhost:11434", help="Ollama host URL"),
    backend: str = typer.Option("ollama", help="Judge backend: ollama | anthropic | openai"),
    min_len: int = typer.Option(50, help="[heuristic] Min assistant response length"),
    max_len: int = typer.Option(4000, help="[heuristic] Max assistant response length"),
    rescore: bool = typer.Option(False, "--rescore", help="Re-score already scored interactions"),
):
    """Score interactions. Scorers: heuristic (fast, no deps) | llm-judge (requires Ollama/API) | ifd (requires Ollama) | combined."""
    from sieve.models import Interaction
    from sieve.score.heuristic import HeuristicScorer

    session = _session()

    if scorer == "heuristic":
        _scorer = HeuristicScorer(min_len, max_len)
        scorer_name = "heuristic"

    elif scorer == "llm-judge":
        if backend == "anthropic":
            from sieve.score.llm_judge import AnthropicJudgeScorer
            _scorer = AnthropicJudgeScorer()
        elif backend == "openai":
            from sieve.score.llm_judge import OpenAIJudgeScorer
            _scorer = OpenAIJudgeScorer()
        else:
            from sieve.score.llm_judge import OllamaJudgeScorer
            _scorer = OllamaJudgeScorer(model=model, host=ollama_host)
        scorer_name = f"llm-judge:{backend}:{model}"

    elif scorer == "ifd":
        from sieve.score.ifd import IFDScorer
        _scorer = IFDScorer(model=model, host=ollama_host)
        scorer_name = f"ifd:{model}"

    elif scorer == "combined":
        from sieve.score.llm_judge import OllamaJudgeScorer
        from sieve.score.ifd import IFDScorer
        from sieve.score.combined import CombinedScorer
        _scorer = CombinedScorer(
            judge_scorer=OllamaJudgeScorer(model=model, host=ollama_host),
            ifd_scorer=IFDScorer(model=model, host=ollama_host),
        )
        scorer_name = f"combined:{model}"

    else:
        console.print(f"[red]Unknown scorer: {scorer}. Use: heuristic | llm-judge | ifd | combined[/red]")
        raise typer.Exit(1)

    query = session.query(Interaction)
    if not rescore:
        query = query.filter(Interaction.quality_score.is_(None))
    interactions = query.all()

    if not interactions:
        console.print("[yellow]No unscored interactions. Use --rescore to re-score.[/yellow]")
        return

    with console.status(f"[cyan]Scoring {len(interactions)} interactions with {scorer}…[/cyan]"):
        for i in interactions:
            s = _scorer.score(i)
            i.quality_score = s
            i.scored_at = datetime.utcnow()
            i.scorer = scorer_name

            # store sub-scores separately when available
            if scorer == "heuristic":
                i.heuristic_score = s
            elif scorer == "llm-judge":
                i.judge_score = s
            elif scorer == "ifd":
                i.ifd_score = s
            elif scorer == "combined":
                i.heuristic_score = HeuristicScorer(min_len, max_len).score(i)

    session.commit()
    console.print(f"[green]✓[/green] Scored [bold]{len(interactions)}[/bold] interactions with [bold]{scorer_name}[/bold]")


# ─── DEDUP ─────────────────────────────────────────────────────────────────────

@app.command()
def dedup(
    threshold: float = typer.Option(0.92, help="Cosine similarity threshold (0–1). Higher = stricter."),
    embed_model: str = typer.Option("all-MiniLM-L6-v2", help="Sentence-transformers model for embeddings"),
    batch_size: int = typer.Option(64, help="Embedding batch size"),
):
    """Semantic deduplication — mark near-duplicate interactions.

    Based on SemDeDup (Meta AI, 2023): removes ~50% redundant data with minimal quality loss.
    Requires: pip install 'trainsieve[embeddings]'
    """
    from sieve.dedup.semantic import run_semantic_dedup

    session = _session()
    with console.status("[cyan]Loading embeddings model and computing similarities…[/cyan]"):
        result = run_semantic_dedup(session, threshold=threshold, embed_model=embed_model, batch_size=batch_size)

    console.print(f"[green]✓[/green] Dedup complete")
    table = Table()
    table.add_column("Metric", style="bold")
    table.add_column("Count")
    table.add_row("Total processed", str(result["total"]))
    table.add_row("Duplicates marked", str(result["duplicates_marked"]))
    table.add_row("Unique kept", str(result["unique_kept"]))
    pct = result["duplicates_marked"] / max(result["total"], 1) * 100
    table.add_row("Reduction", f"{pct:.1f}%")
    console.print(table)


# ─── STATS ─────────────────────────────────────────────────────────────────────

@app.command()
def stats():
    """Show pipeline stats."""
    from sqlalchemy import func
    from sieve.models import Interaction, DatasetVersion, TrainingRun

    session = _session()
    total = session.query(func.count(Interaction.id)).scalar()
    scored = session.query(func.count(Interaction.id)).filter(Interaction.quality_score.isnot(None)).scalar()
    avg = session.query(func.avg(Interaction.quality_score)).filter(Interaction.quality_score.isnot(None)).scalar()
    dups = session.query(func.count(Interaction.id)).filter(Interaction.is_duplicate.isnot(None)).scalar()
    versions = session.query(func.count(DatasetVersion.id)).scalar()
    runs = session.query(func.count(TrainingRun.id)).scalar()

    table = Table(title="Sieve Pipeline Stats")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Total interactions", str(total))
    table.add_row("Scored", str(scored))
    table.add_row("Avg quality score", f"{avg:.3f}" if avg else "—")
    table.add_row("Duplicates marked", str(dups))
    table.add_row("Dataset versions", str(versions))
    table.add_row("Training runs", str(runs))
    console.print(table)


# ─── DATASET ───────────────────────────────────────────────────────────────────

@dataset_app.command("create")
def dataset_create(
    name: str = typer.Argument(..., help="Version name, e.g. v1.0"),
    min_quality: float = typer.Option(0.6, help="Minimum quality score filter"),
    description: Optional[str] = typer.Option(None, help="Description"),
    diff: Optional[str] = typer.Option(None, help="Parent version — only include new interactions"),
    exclude_duplicates: bool = typer.Option(True, help="Exclude interactions marked as duplicates"),
):
    """Create a dataset version from scored interactions."""
    from sieve.curate.dataset import DatasetManager

    session = _session()
    mgr = DatasetManager(session)
    version = mgr.create_version(name, min_quality, description, diff, exclude_duplicates)
    console.print(
        f"[green]✓[/green] Dataset [bold]{name!r}[/bold] created "
        f"with [bold]{len(version.interactions)}[/bold] interactions "
        f"(min_quality={min_quality})"
    )


@dataset_app.command("export")
def dataset_export(
    name: str = typer.Argument(..., help="Version name to export"),
    output: Path = typer.Option(Path("./export.jsonl"), help="Output JSONL path"),
):
    """Export a dataset version to JSONL for fine-tuning."""
    from sieve.curate.dataset import DatasetManager

    session = _session()
    count = DatasetManager(session).export_jsonl(name, output)
    console.print(f"[green]✓[/green] Exported [bold]{count}[/bold] interactions to {output}")


@dataset_app.command("list")
def dataset_list():
    """List all dataset versions."""
    from sieve.models import DatasetVersion

    session = _session()
    versions = session.query(DatasetVersion).order_by(DatasetVersion.created_at).all()

    if not versions:
        console.print("[yellow]No dataset versions yet. Run: sieve dataset create[/yellow]")
        return

    table = Table(title="Dataset Versions")
    table.add_column("Name", style="bold")
    table.add_column("Interactions")
    table.add_column("Min Quality")
    table.add_column("Parent")
    table.add_column("Created")
    for v in versions:
        table.add_row(
            v.name,
            str(len(v.interactions)),
            f"{v.min_quality_score:.2f}" if v.min_quality_score is not None else "—",
            v.parent_name or "—",
            v.created_at.strftime("%Y-%m-%d %H:%M"),
        )
    console.print(table)


# ─── TRAIN ─────────────────────────────────────────────────────────────────────

@app.command()
def train(
    version: str = typer.Argument(..., help="Dataset version name"),
    base_model: str = typer.Option("meta-llama/Llama-3.2-3B-Instruct", help="HuggingFace model ID"),
    output_dir: Path = typer.Option(Path("./output"), help="Training output directory"),
    export_path: Path = typer.Option(Path("./export.jsonl"), help="Path to exported JSONL dataset"),
):
    """Generate Axolotl config and show the training command."""
    from sieve.trigger.axolotl import AxolotlTrigger

    session = _session()
    run = AxolotlTrigger(session).trigger(version, export_path, base_model, output_dir)
    config_path = run.config["config_path"]

    console.print(f"[green]✓[/green] Training run recorded [dim](id: {run.id[:8]})[/dim]")
    console.print(f"\n[bold]Run training with Axolotl:[/bold]")
    console.print(f"  pip install axolotl")
    console.print(f"  axolotl train {config_path}")


if __name__ == "__main__":
    app()
