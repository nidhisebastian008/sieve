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


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="JSONL file to ingest"),
):
    """Ingest interactions from a JSONL file."""
    from sieve.ingest.jsonl import JSONLIngester

    session = _session()
    ingester = JSONLIngester(path)
    count = 0
    for interaction in ingester.ingest():
        session.add(interaction)
        count += 1
    session.commit()
    console.print(f"[green]✓[/green] Ingested [bold]{count}[/bold] interactions from {path}")


@app.command()
def score(
    min_len: int = typer.Option(50, help="Min assistant response character length"),
    max_len: int = typer.Option(4000, help="Max assistant response character length"),
    rescore: bool = typer.Option(False, "--rescore", help="Re-score already scored interactions"),
):
    """Score interactions with heuristic scorer."""
    from sieve.models import Interaction
    from sieve.score.heuristic import HeuristicScorer

    session = _session()
    scorer = HeuristicScorer(min_len, max_len)

    query = session.query(Interaction)
    if not rescore:
        query = query.filter(Interaction.quality_score.is_(None))

    interactions = query.all()
    for i in interactions:
        i.quality_score = scorer.score(i)
        i.scored_at = datetime.utcnow()
        i.scorer = "heuristic"
    session.commit()
    console.print(f"[green]✓[/green] Scored [bold]{len(interactions)}[/bold] interactions")


@app.command()
def stats():
    """Show pipeline stats."""
    from sqlalchemy import func
    from sieve.models import Interaction, DatasetVersion, TrainingRun

    session = _session()

    total = session.query(func.count(Interaction.id)).scalar()
    scored = session.query(func.count(Interaction.id)).filter(Interaction.quality_score.isnot(None)).scalar()
    avg = session.query(func.avg(Interaction.quality_score)).filter(Interaction.quality_score.isnot(None)).scalar()
    versions = session.query(func.count(DatasetVersion.id)).scalar()
    runs = session.query(func.count(TrainingRun.id)).scalar()

    table = Table(title="Sieve Pipeline Stats")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Total interactions", str(total))
    table.add_row("Scored", str(scored))
    table.add_row("Avg quality score", f"{avg:.3f}" if avg else "—")
    table.add_row("Dataset versions", str(versions))
    table.add_row("Training runs", str(runs))
    console.print(table)


@dataset_app.command("create")
def dataset_create(
    name: str = typer.Argument(..., help="Version name, e.g. v1.0"),
    min_quality: float = typer.Option(0.6, help="Minimum quality score filter"),
    description: Optional[str] = typer.Option(None, help="Description"),
    diff: Optional[str] = typer.Option(None, help="Parent version — only include new interactions"),
):
    """Create a dataset version from scored interactions."""
    from sieve.curate.dataset import DatasetManager

    session = _session()
    mgr = DatasetManager(session)
    version = mgr.create_version(name, min_quality, description, diff)
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
    mgr = DatasetManager(session)
    count = mgr.export_jsonl(name, output)
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
    trigger = AxolotlTrigger(session)
    run = trigger.trigger(version, export_path, base_model, output_dir)
    config_path = run.config["config_path"]

    console.print(f"[green]✓[/green] Training run recorded [dim](id: {run.id[:8]})[/dim]")
    console.print(f"\n[bold]Run training with Axolotl:[/bold]")
    console.print(f"  pip install axolotl")
    console.print(f"  axolotl train {config_path}")


if __name__ == "__main__":
    app()
