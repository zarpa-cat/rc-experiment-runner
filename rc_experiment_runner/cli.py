"""CLI interface for the experiment runner."""

from datetime import UTC, datetime
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from rc_experiment_runner.models import Experiment, Variant
from rc_experiment_runner.runner import ExperimentRunner

app = typer.Typer(name="rce", help="RevenueCat Experiment Runner CLI")
console = Console()

DbPathOption = Annotated[
    str, typer.Option("--db-path", help="Path to the SQLite database file")
]


def _get_runner(db_path: str) -> ExperimentRunner:
    return ExperimentRunner(db_path=db_path)


def _parse_variant(variant_str: str) -> tuple[str, float]:
    """Parse a variant string like 'control:0.5' into (id, weight)."""
    parts = variant_str.split(":")
    if len(parts) != 2:
        raise typer.BadParameter(f"Variant must be in format 'id:weight', got '{variant_str}'")
    try:
        weight = float(parts[1])
    except ValueError:
        raise typer.BadParameter(f"Invalid weight '{parts[1]}' in variant '{variant_str}'")
    return parts[0], weight


@app.command()
def create(
    experiment_id: str,
    name: str,
    variant: Annotated[
        list[str], typer.Option("--variant", "-v", help="Variant in format 'id:weight'")
    ],
    description: Annotated[str, typer.Option("--description", "-d")] = "",
    end_date: Annotated[str | None, typer.Option("--end-date")] = None,
    salt: Annotated[str, typer.Option("--salt")] = "",
    db_path: DbPathOption = "experiments.db",
) -> None:
    """Create a new experiment."""
    variants = []
    for v in variant:
        vid, weight = _parse_variant(v)
        variants.append(Variant(id=vid, name=vid, description=description, weight=weight))

    parsed_end_date = None
    if end_date:
        parsed_end_date = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)

    experiment = Experiment(
        id=experiment_id,
        name=name,
        variants=variants,
        start_date=datetime.now(UTC),
        end_date=parsed_end_date,
        salt=salt,
    )

    runner = _get_runner(db_path)
    runner.create_experiment(experiment)
    msg = f"Created experiment '{experiment_id}' with {len(variants)} variants."
    console.print(f"[green]{msg}[/green]")


@app.command("list")
def list_experiments(
    db_path: DbPathOption = "experiments.db",
) -> None:
    """List all experiments."""
    runner = _get_runner(db_path)
    experiments = runner.list_experiments()

    if not experiments:
        console.print("[yellow]No experiments found.[/yellow]")
        return

    table = Table(title="Experiments")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Variants", justify="right")
    table.add_column("Status", style="bold")
    table.add_column("Start Date")
    table.add_column("End Date")

    now = datetime.now(UTC)
    for exp in experiments:
        if exp.end_date and exp.end_date < now:
            status = "[red]ended[/red]"
        elif exp.start_date > now:
            status = "[yellow]upcoming[/yellow]"
        else:
            status = "[green]active[/green]"

        table.add_row(
            exp.id,
            exp.name,
            str(len(exp.variants)),
            status,
            exp.start_date.strftime("%Y-%m-%d"),
            exp.end_date.strftime("%Y-%m-%d") if exp.end_date else "—",
        )

    console.print(table)


@app.command()
def assign(
    subscriber_id: str,
    experiment_id: str,
    db_path: DbPathOption = "experiments.db",
) -> None:
    """Assign a subscriber to a variant."""
    runner = _get_runner(db_path)
    variant = runner.assign(subscriber_id, experiment_id)
    console.print(
        f"Subscriber [cyan]{subscriber_id}[/cyan] "
        f"→ variant [bold]{variant.id}[/bold] ({variant.name})"
    )


@app.command()
def convert(
    subscriber_id: str,
    experiment_id: str,
    value: Annotated[float, typer.Option("--value")] = 0.0,
    event_type: Annotated[str, typer.Option("--event-type")] = "conversion",
    db_path: DbPathOption = "experiments.db",
) -> None:
    """Record a conversion event."""
    runner = _get_runner(db_path)
    runner.record_conversion(subscriber_id, experiment_id, event_type=event_type, value=value)
    console.print(f"[green]Recorded conversion for {subscriber_id} in {experiment_id}[/green]")


@app.command()
def results(
    experiment_id: str,
    db_path: DbPathOption = "experiments.db",
) -> None:
    """Show experiment results."""
    runner = _get_runner(db_path)
    res = runner.results(experiment_id)

    table = Table(title=f"Results: {experiment_id}")
    table.add_column("Variant", style="cyan")
    table.add_column("Assignments", justify="right")
    table.add_column("Conversions", justify="right")
    table.add_column("Rate", justify="right")
    table.add_column("Avg Value", justify="right")

    for stats in res.variant_stats.values():
        table.add_row(
            stats.variant_id,
            str(stats.assignments),
            str(stats.conversions),
            f"{stats.conversion_rate:.2%}",
            f"{stats.avg_value:.2f}",
        )

    console.print(table)
    console.print(f"Total subjects: {res.total_subjects}")


@app.command()
def delete(
    experiment_id: str,
    db_path: DbPathOption = "experiments.db",
) -> None:
    """Delete an experiment and all its data."""
    runner = _get_runner(db_path)
    runner._store.delete_experiment(experiment_id)
    console.print(f"[red]Deleted experiment '{experiment_id}'[/red]")
