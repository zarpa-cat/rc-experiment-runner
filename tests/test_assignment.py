"""Tests for deterministic variant assignment."""

from datetime import UTC, datetime

from rc_experiment_runner.assignment import assign_variant
from rc_experiment_runner.models import Experiment, Variant


def _make_experiment(
    experiment_id: str = "exp-1",
    variants: list[Variant] | None = None,
    salt: str = "",
) -> Experiment:
    if variants is None:
        variants = [
            Variant(id="control", name="Control", weight=0.5),
            Variant(id="treatment", name="Treatment", weight=0.5),
        ]
    return Experiment(
        id=experiment_id,
        name="Test Experiment",
        variants=variants,
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        salt=salt,
    )


class TestAssignmentDeterminism:
    def test_same_subscriber_same_experiment_same_result(self):
        exp = _make_experiment()
        v1 = assign_variant("user-123", exp)
        v2 = assign_variant("user-123", exp)
        assert v1.id == v2.id

    def test_deterministic_across_calls(self):
        exp = _make_experiment()
        results = [assign_variant("user-abc", exp).id for _ in range(100)]
        assert len(set(results)) == 1

    def test_different_subscribers_can_get_different_variants(self):
        exp = _make_experiment()
        variants = {assign_variant(f"user-{i}", exp).id for i in range(200)}
        assert len(variants) == 2  # Both variants should appear

    def test_different_experiments_different_assignments(self):
        exp1 = _make_experiment(experiment_id="exp-1")
        exp2 = _make_experiment(experiment_id="exp-2")
        # At least some users should get different assignments across experiments
        differences = 0
        for i in range(100):
            v1 = assign_variant(f"user-{i}", exp1)
            v2 = assign_variant(f"user-{i}", exp2)
            if v1.id != v2.id:
                differences += 1
        assert differences > 0


class TestAssignmentWeights:
    def test_equal_weights_roughly_balanced(self):
        exp = _make_experiment()
        counts = {"control": 0, "treatment": 0}
        n = 10000
        for i in range(n):
            v = assign_variant(f"user-{i}", exp)
            counts[v.id] += 1
        # Should be roughly 50/50, allow 5% margin
        assert abs(counts["control"] / n - 0.5) < 0.05
        assert abs(counts["treatment"] / n - 0.5) < 0.05

    def test_unequal_weights_respected(self):
        exp = _make_experiment(
            variants=[
                Variant(id="control", name="Control", weight=0.8),
                Variant(id="treatment", name="Treatment", weight=0.2),
            ]
        )
        counts = {"control": 0, "treatment": 0}
        n = 10000
        for i in range(n):
            v = assign_variant(f"user-{i}", exp)
            counts[v.id] += 1
        assert abs(counts["control"] / n - 0.8) < 0.05
        assert abs(counts["treatment"] / n - 0.2) < 0.05

    def test_three_way_split(self):
        exp = _make_experiment(
            variants=[
                Variant(id="a", name="A", weight=0.5),
                Variant(id="b", name="B", weight=0.3),
                Variant(id="c", name="C", weight=0.2),
            ]
        )
        counts = {"a": 0, "b": 0, "c": 0}
        n = 10000
        for i in range(n):
            v = assign_variant(f"user-{i}", exp)
            counts[v.id] += 1
        assert abs(counts["a"] / n - 0.5) < 0.05
        assert abs(counts["b"] / n - 0.3) < 0.05
        assert abs(counts["c"] / n - 0.2) < 0.05

    def test_100_percent_single_variant(self):
        exp = _make_experiment(
            variants=[Variant(id="only", name="Only", weight=1.0)]
        )
        for i in range(100):
            v = assign_variant(f"user-{i}", exp)
            assert v.id == "only"


class TestAssignmentSalt:
    def test_salt_changes_assignments(self):
        exp_no_salt = _make_experiment(salt="")
        exp_salted = _make_experiment(salt="my-salt")
        differences = 0
        for i in range(100):
            v1 = assign_variant(f"user-{i}", exp_no_salt)
            v2 = assign_variant(f"user-{i}", exp_salted)
            if v1.id != v2.id:
                differences += 1
        assert differences > 0

    def test_same_salt_same_result(self):
        exp1 = _make_experiment(salt="salt-a")
        exp2 = _make_experiment(salt="salt-a")
        for i in range(50):
            v1 = assign_variant(f"user-{i}", exp1)
            v2 = assign_variant(f"user-{i}", exp2)
            assert v1.id == v2.id
