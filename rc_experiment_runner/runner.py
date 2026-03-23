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
from rc_experiment_runner.rc_client import RCClient
from rc_experiment_runner.store import ExperimentStore

# Attribute name written to RC subscriber profile on assignment
RC_EXPERIMENT_ATTR_PREFIX = "rce_experiment_"


class ExperimentRunner:
    """High-level experiment runner that coordinates assignment, storage, and results."""

    def __init__(
        self,
        db_path: str = "experiments.db",
        rc_client: RCClient | None = None,
    ) -> None:
        self._store = ExperimentStore(db_path)
        self._rc = rc_client

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

    async def assign_with_rc_sync(
        self,
        subscriber_id: str,
        experiment_id: str,
        offering_map: dict[str, str] | None = None,
    ) -> Variant:
        """Assign a subscriber and sync the assignment to RevenueCat.

        After assignment, writes two RC subscriber attributes:
          - rce_experiment_{experiment_id} = variant_id
          - rce_variant_{experiment_id} = variant_id  (alias for easy segmentation)

        If offering_map is provided (variant_id → offering_id), also overrides
        the subscriber's active RC offering to match their assigned variant.

        Requires a configured RCClient. Falls back to local-only assignment if
        RC is not configured (no exception raised).
        """
        variant = self.assign(subscriber_id, experiment_id)

        if self._rc is not None:
            attr_key = f"{RC_EXPERIMENT_ATTR_PREFIX}{experiment_id}"
            await self._rc.set_subscriber_attribute(subscriber_id, attr_key, variant.id)

            if offering_map and variant.id in offering_map:
                offering_id = offering_map[variant.id]
                await self._rc.set_active_offering(subscriber_id, offering_id)

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
