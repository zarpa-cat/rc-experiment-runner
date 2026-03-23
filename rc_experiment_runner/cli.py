"""CLI interface for the experiment runner."""

from datetime import UTC, datetime
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rc_experiment_runner.analysis import build_report, detect_winner, z_test_proportions
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
def analyze(
    experiment_id: str,
    control: Annotated[str | None, typer.Option("--control", "-c", help="Control variant ID")] = (
        None
    ),
    confidence: Annotated[float, typer.Option("--confidence", help="Confidence level (0-1)")] = (
        0.95
    ),
    db_path: DbPathOption = "experiments.db",
) -> None:
    """Run statistical analysis comparing variants against the control.

    For each non-control variant, runs a two-proportion z-test and prints
    the z-score, p-value, confidence intervals, and whether the result is
    statistically significant.
    """
    runner = _get_runner(db_path)
    res = runner.results(experiment_id)
    experiment = runner._store.get_experiment(experiment_id)
    if experiment is None:
        console.print(f"[red]Experiment '{experiment_id}' not found.[/red]")
        raise typer.Exit(1)

    # Resolve control
    control_id = control or experiment.variants[0].id
    if control_id not in res.variant_stats:
        console.print(f"[red]Control variant '{control_id}' not found in results.[/red]")
        raise typer.Exit(1)

    control_stats = res.variant_stats[control_id]
    alpha = 1.0 - confidence

    table = Table(title=f"Statistical Analysis: {experiment_id}  (α={alpha:.2f})")
    table.add_column("Comparison", style="cyan")
    table.add_column("Control Rate", justify="right")
    table.add_column("Treatment Rate", justify="right")
    table.add_column("Uplift", justify="right")
    table.add_column("95% CI (treatment)", justify="right")
    table.add_column("Z-score", justify="right")
    table.add_column("P-value", justify="right")
    table.add_column("Significant?", justify="center")

    any_results = False
    for vid, stats in res.variant_stats.items():
        if vid == control_id:
            continue
        any_results = True
        result = z_test_proportions(
            control_conversions=control_stats.conversions,
            control_n=control_stats.assignments,
            treatment_conversions=stats.conversions,
            treatment_n=stats.assignments,
            confidence_level=confidence,
        )
        result = result.model_copy(update={"control_id": control_id, "treatment_id": vid})

        uplift_str = (
            f"{result.relative_uplift:+.1%}"
            if result.relative_uplift != float("inf")
            else "∞"
        )
        sig_cell = "[green]✓ YES[/green]" if result.is_significant else "[yellow]✗ NO[/yellow]"
        ci_str = f"[{result.treatment_ci_lower:.3f}, {result.treatment_ci_upper:.3f}]"

        table.add_row(
            f"{control_id} → {vid}",
            f"{result.control_rate:.3f}",
            f"{result.treatment_rate:.3f}",
            uplift_str,
            ci_str,
            f"{result.z_score:.3f}",
            f"{result.p_value:.4f}",
            sig_cell,
        )

    if not any_results:
        console.print("[yellow]Only one variant — nothing to compare.[/yellow]")
        return

    console.print(table)

    # Winner summary
    winner = detect_winner(res, experiment, control_id, confidence)
    if winner.winner_id:
        console.print(
            f"\n🏆 [bold green]Winner: {winner.winner_id}[/bold green] "
            f"(significant at {confidence:.0%} confidence)"
        )
    else:
        console.print(
            f"\n[yellow]No significant winner yet at {confidence:.0%} confidence.[/yellow]"
        )


@app.command()
def report(
    experiment_id: str,
    control: Annotated[str | None, typer.Option("--control", "-c")] = None,
    confidence: Annotated[float, typer.Option("--confidence")] = 0.95,
    db_path: DbPathOption = "experiments.db",
) -> None:
    """Print a full experiment report: results + statistical analysis + winner."""
    runner = _get_runner(db_path)
    res = runner.results(experiment_id)
    experiment = runner._store.get_experiment(experiment_id)
    if experiment is None:
        console.print(f"[red]Experiment '{experiment_id}' not found.[/red]")
        raise typer.Exit(1)

    rpt = build_report(res, experiment, control_id=control, confidence_level=confidence)

    # Header
    console.print(
        Panel(
            f"[bold]{rpt.experiment_name}[/bold]\n"
            f"ID: {rpt.experiment_id}  |  "
            f"Subjects: {rpt.total_subjects}  |  "
            f"Generated: {rpt.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            title="Experiment Report",
        )
    )

    # Results table
    res_table = Table(title="Variant Results")
    res_table.add_column("Variant", style="cyan")
    res_table.add_column("Assignments", justify="right")
    res_table.add_column("Conversions", justify="right")
    res_table.add_column("Rate", justify="right")
    res_table.add_column("Total Value", justify="right")
    res_table.add_column("Avg Value", justify="right")

    for vid, stats in rpt.variant_stats.items():
        res_table.add_row(
            vid,
            str(stats.assignments),
            str(stats.conversions),
            f"{stats.conversion_rate:.3f}",
            f"{stats.total_value:.2f}",
            f"{stats.avg_value:.2f}",
        )
    console.print(res_table)

    # Statistical comparisons
    if rpt.winner.comparisons:
        stat_table = Table(title=f"Statistical Comparisons (confidence={confidence:.0%})")
        stat_table.add_column("Treatment", style="cyan")
        stat_table.add_column("Uplift", justify="right")
        stat_table.add_column("Z-score", justify="right")
        stat_table.add_column("P-value", justify="right")
        stat_table.add_column("Significant", justify="center")

        for cmp in rpt.winner.comparisons:
            uplift_str = (
                f"{cmp.relative_uplift:+.1%}" if cmp.relative_uplift != float("inf") else "∞"
            )
            sig = "[green]✓[/green]" if cmp.is_significant else "[yellow]✗[/yellow]"
            stat_table.add_row(
                cmp.treatment_id,
                uplift_str,
                f"{cmp.z_score:.3f}",
                f"{cmp.p_value:.4f}",
                sig,
            )
        console.print(stat_table)

    # Winner verdict
    if rpt.winner.winner_id:
        console.print(
            f"\n🏆 [bold green]Winner: {rpt.winner.winner_id}[/bold green] "
            f"(statistically significant at {confidence:.0%})"
        )
    else:
        console.print(
            f"\n[yellow]⏳ No significant winner yet at {confidence:.0%} confidence.[/yellow]"
        )


@app.command()
def delete(
    experiment_id: str,
    db_path: DbPathOption = "experiments.db",
) -> None:
    """Delete an experiment and all its data."""
    runner = _get_runner(db_path)
    runner._store.delete_experiment(experiment_id)
    console.print(f"[red]Deleted experiment '{experiment_id}'[/red]")
