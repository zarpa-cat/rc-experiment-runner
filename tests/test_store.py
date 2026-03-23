"""Tests for SQLite-backed experiment store."""

import os
import tempfile
from datetime import UTC, datetime

import pytest

from rc_experiment_runner.models import (
    Assignment,
    ConversionEvent,
    Experiment,
    Variant,
)
from rc_experiment_runner.store import ExperimentStore


@pytest.fixture
def store():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    s = ExperimentStore(db_path)
    yield s
    os.unlink(db_path)


def _make_experiment(experiment_id: str = "exp-1") -> Experiment:
    return Experiment(
        id=experiment_id,
        name="Pricing Test",
        variants=[
            Variant(id="control", name="Control", weight=0.5),
            Variant(id="treatment", name="Treatment", weight=0.5),
        ],
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime(2024, 3, 1, tzinfo=UTC),
        salt="test-salt",
    )


class TestExperimentCrud:
    def test_create_and_get(self, store):
        exp = _make_experiment()
        store.create_experiment(exp)
        retrieved = store.get_experiment("exp-1")
        assert retrieved is not None
        assert retrieved.id == "exp-1"
        assert retrieved.name == "Pricing Test"
        assert len(retrieved.variants) == 2
        assert retrieved.salt == "test-salt"

    def test_get_nonexistent(self, store):
        assert store.get_experiment("nope") is None

    def test_list_experiments(self, store):
        store.create_experiment(_make_experiment("exp-1"))
        store.create_experiment(_make_experiment("exp-2"))
        experiments = store.list_experiments()
        assert len(experiments) == 2
        assert {e.id for e in experiments} == {"exp-1", "exp-2"}

    def test_list_empty(self, store):
        assert store.list_experiments() == []

    def test_delete_experiment(self, store):
        store.create_experiment(_make_experiment())
        store.delete_experiment("exp-1")
        assert store.get_experiment("exp-1") is None

    def test_delete_removes_assignments_and_conversions(self, store):
        exp = _make_experiment()
        store.create_experiment(exp)
        store.record_assignment(Assignment(
            subscriber_id="user-1",
            experiment_id="exp-1",
            variant_id="control",
            assigned_at=datetime.now(UTC),
        ))
        store.record_conversion(ConversionEvent(
            subscriber_id="user-1",
            experiment_id="exp-1",
            event_type="conversion",
            value=9.99,
            recorded_at=datetime.now(UTC),
        ))
        store.delete_experiment("exp-1")
        assert store.get_assignment("user-1", "exp-1") is None

    def test_experiment_preserves_variant_data(self, store):
        exp = _make_experiment()
        store.create_experiment(exp)
        retrieved = store.get_experiment("exp-1")
        assert retrieved.variants[0].id == "control"
        assert retrieved.variants[0].weight == 0.5
        assert retrieved.variants[1].id == "treatment"

    def test_experiment_without_end_date(self, store):
        exp = Experiment(
            id="no-end",
            name="Open-ended",
            variants=[Variant(id="a", name="A", weight=1.0)],
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
        )
        store.create_experiment(exp)
        retrieved = store.get_experiment("no-end")
        assert retrieved.end_date is None


class TestAssignmentStorage:
    def test_record_and_get_assignment(self, store):
        store.create_experiment(_make_experiment())
        assignment = Assignment(
            subscriber_id="user-1",
            experiment_id="exp-1",
            variant_id="control",
            assigned_at=datetime(2024, 1, 15, tzinfo=UTC),
        )
        store.record_assignment(assignment)
        retrieved = store.get_assignment("user-1", "exp-1")
        assert retrieved is not None
        assert retrieved.variant_id == "control"
        assert retrieved.subscriber_id == "user-1"

    def test_idempotent_assignment(self, store):
        store.create_experiment(_make_experiment())
        a1 = Assignment(
            subscriber_id="user-1",
            experiment_id="exp-1",
            variant_id="control",
            assigned_at=datetime(2024, 1, 15, tzinfo=UTC),
        )
        a2 = Assignment(
            subscriber_id="user-1",
            experiment_id="exp-1",
            variant_id="treatment",  # Different variant — should be ignored
            assigned_at=datetime(2024, 2, 1, tzinfo=UTC),
        )
        store.record_assignment(a1)
        store.record_assignment(a2)
        retrieved = store.get_assignment("user-1", "exp-1")
        assert retrieved.variant_id == "control"  # First assignment wins

    def test_get_nonexistent_assignment(self, store):
        assert store.get_assignment("user-1", "exp-1") is None


class TestConversionsAndResults:
    def test_record_conversion(self, store):
        store.create_experiment(_make_experiment())
        store.record_assignment(Assignment(
            subscriber_id="user-1",
            experiment_id="exp-1",
            variant_id="control",
            assigned_at=datetime.now(UTC),
        ))
        store.record_conversion(ConversionEvent(
            subscriber_id="user-1",
            experiment_id="exp-1",
            event_type="purchase",
            value=9.99,
            recorded_at=datetime.now(UTC),
        ))
        results = store.get_results("exp-1")
        assert results.variant_stats["control"].conversions == 1
        assert results.variant_stats["control"].total_value == 9.99

    def test_results_calculation(self, store):
        store.create_experiment(_make_experiment())

        # 3 users in control, 2 convert
        for i in range(3):
            store.record_assignment(Assignment(
                subscriber_id=f"ctrl-{i}",
                experiment_id="exp-1",
                variant_id="control",
                assigned_at=datetime.now(UTC),
            ))
        for i in range(2):
            store.record_conversion(ConversionEvent(
                subscriber_id=f"ctrl-{i}",
                experiment_id="exp-1",
                value=10.0,
                recorded_at=datetime.now(UTC),
            ))

        # 2 users in treatment, 1 converts
        for i in range(2):
            store.record_assignment(Assignment(
                subscriber_id=f"treat-{i}",
                experiment_id="exp-1",
                variant_id="treatment",
                assigned_at=datetime.now(UTC),
            ))
        store.record_conversion(ConversionEvent(
            subscriber_id="treat-0",
            experiment_id="exp-1",
            value=20.0,
            recorded_at=datetime.now(UTC),
        ))

        results = store.get_results("exp-1")
        assert results.total_subjects == 5

        ctrl = results.variant_stats["control"]
        assert ctrl.assignments == 3
        assert ctrl.conversions == 2
        assert abs(ctrl.conversion_rate - 2 / 3) < 1e-6
        assert ctrl.total_value == 20.0
        assert ctrl.avg_value == 10.0

        treat = results.variant_stats["treatment"]
        assert treat.assignments == 2
        assert treat.conversions == 1
        assert treat.conversion_rate == 0.5
        assert treat.total_value == 20.0
        assert treat.avg_value == 20.0

    def test_results_no_conversions(self, store):
        store.create_experiment(_make_experiment())
        store.record_assignment(Assignment(
            subscriber_id="user-1",
            experiment_id="exp-1",
            variant_id="control",
            assigned_at=datetime.now(UTC),
        ))
        results = store.get_results("exp-1")
        assert results.variant_stats["control"].conversions == 0
        assert results.variant_stats["control"].conversion_rate == 0.0
        assert results.variant_stats["control"].avg_value == 0.0

    def test_results_nonexistent_experiment(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.get_results("nope")

    def test_multiple_conversions_per_user(self, store):
        store.create_experiment(_make_experiment())
        store.record_assignment(Assignment(
            subscriber_id="user-1",
            experiment_id="exp-1",
            variant_id="control",
            assigned_at=datetime.now(UTC),
        ))
        for _ in range(3):
            store.record_conversion(ConversionEvent(
                subscriber_id="user-1",
                experiment_id="exp-1",
                value=5.0,
                recorded_at=datetime.now(UTC),
            ))
        results = store.get_results("exp-1")
        assert results.variant_stats["control"].conversions == 3
        assert results.variant_stats["control"].total_value == 15.0
