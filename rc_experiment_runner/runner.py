"""High-level API for running experiments."""

from datetime import UTC, datetime

from rc_experiment_runner.assignment import assign_variant
from rc_experiment_runner.models import (
    Assignment,
    ConversionEvent,
    Experiment,
    ExperimentResults,
    Variant,
)
from rc_experiment_runner.store import ExperimentStore


class ExperimentRunner:
    """High-level experiment runner that coordinates assignment, storage, and results."""

    def __init__(self, db_path: str = "experiments.db") -> None:
        self._store = ExperimentStore(db_path)

    def create_experiment(self, experiment: Experiment) -> None:
        """Create and persist a new experiment."""
        self._store.create_experiment(experiment)

    def assign(self, subscriber_id: str, experiment_id: str) -> Variant:
        """Assign a subscriber to a variant, or return existing assignment.

        If the subscriber already has an assignment for this experiment,
        returns the previously assigned variant. Otherwise, deterministically
        assigns a variant and records it.
        """
        existing = self._store.get_assignment(subscriber_id, experiment_id)
        if existing is not None:
            experiment = self._store.get_experiment(experiment_id)
            if experiment is None:
                raise ValueError(f"Experiment '{experiment_id}' not found")
            for v in experiment.variants:
                if v.id == existing.variant_id:
                    return v
            raise ValueError(f"Variant '{existing.variant_id}' not found in experiment")

        experiment = self._store.get_experiment(experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        variant = assign_variant(subscriber_id, experiment)
        assignment = Assignment(
            subscriber_id=subscriber_id,
            experiment_id=experiment_id,
            variant_id=variant.id,
            assigned_at=datetime.now(UTC),
        )
        self._store.record_assignment(assignment)
        return variant

    def record_conversion(
        self,
        subscriber_id: str,
        experiment_id: str,
        event_type: str = "conversion",
        value: float = 0.0,
    ) -> None:
        """Record a conversion event for a subscriber."""
        event = ConversionEvent(
            subscriber_id=subscriber_id,
            experiment_id=experiment_id,
            event_type=event_type,
            value=value,
            recorded_at=datetime.now(UTC),
        )
        self._store.record_conversion(event)

    def results(self, experiment_id: str) -> ExperimentResults:
        """Get aggregated results for an experiment."""
        return self._store.get_results(experiment_id)

    def list_experiments(self) -> list[Experiment]:
        """List all experiments."""
        return self._store.list_experiments()
