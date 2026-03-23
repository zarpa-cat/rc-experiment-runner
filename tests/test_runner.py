"""Tests for the high-level ExperimentRunner API."""

import os
import tempfile
from datetime import UTC, datetime

import pytest

from rc_experiment_runner.models import Experiment, Variant
from rc_experiment_runner.runner import ExperimentRunner


@pytest.fixture
def runner():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    r = ExperimentRunner(db_path=db_path)
    yield r
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
    )


class TestRunnerLifecycle:
    def test_create_and_list(self, runner):
        runner.create_experiment(_make_experiment("exp-1"))
        runner.create_experiment(_make_experiment("exp-2"))
        experiments = runner.list_experiments()
        assert len(experiments) == 2

    def test_assign_returns_variant(self, runner):
        runner.create_experiment(_make_experiment())
        variant = runner.assign("user-1", "exp-1")
        assert variant.id in ("control", "treatment")

    def test_assign_is_deterministic(self, runner):
        runner.create_experiment(_make_experiment())
        v1 = runner.assign("user-1", "exp-1")
        v2 = runner.assign("user-1", "exp-1")
        assert v1.id == v2.id

    def test_assign_nonexistent_experiment(self, runner):
        with pytest.raises(ValueError, match="not found"):
            runner.assign("user-1", "nope")

    def test_full_lifecycle(self, runner):
        """Create → assign → convert → results."""
        runner.create_experiment(_make_experiment())

        # Assign 100 users
        assignments = {}
        for i in range(100):
            v = runner.assign(f"user-{i}", "exp-1")
            assignments[f"user-{i}"] = v.id

        # Convert some users
        for uid, vid in list(assignments.items())[:30]:
            runner.record_conversion(uid, "exp-1", value=9.99)

        results = runner.results("exp-1")
        assert results.total_subjects == 100
        total_conversions = sum(s.conversions for s in results.variant_stats.values())
        assert total_conversions == 30

    def test_conversion_records_value(self, runner):
        runner.create_experiment(_make_experiment())
        runner.assign("user-1", "exp-1")
        runner.record_conversion("user-1", "exp-1", value=49.99)
        results = runner.results("exp-1")
        total_value = sum(s.total_value for s in results.variant_stats.values())
        assert total_value == 49.99

    def test_conversion_with_custom_event_type(self, runner):
        runner.create_experiment(_make_experiment())
        runner.assign("user-1", "exp-1")
        runner.record_conversion("user-1", "exp-1", event_type="purchase", value=19.99)
        results = runner.results("exp-1")
        total_conversions = sum(s.conversions for s in results.variant_stats.values())
        assert total_conversions == 1
