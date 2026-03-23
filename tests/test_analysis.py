"""Tests for statistical analysis: Wilson CIs, z-tests, winner detection, reports."""


from datetime import UTC, datetime

import pytest

from rc_experiment_runner.analysis import (
    _norm_ppf,
    _normal_cdf,
    _p_value_two_tailed,
    _z_alpha,
    build_report,
    detect_winner,
    wilson_ci,
    z_test_proportions,
)
from rc_experiment_runner.models import (
    Experiment,
    ExperimentReport,
    ExperimentResults,
    Variant,
    VariantStats,
    WinnerResult,
)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _make_experiment(control_id: str = "control", treatment_id: str = "treatment") -> Experiment:
    return Experiment(
        id="test-exp",
        name="Test Experiment",
        variants=[
            Variant(id=control_id, name="Control", weight=0.5),
            Variant(id=treatment_id, name="Treatment", weight=0.5),
        ],
        start_date=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _make_results(
    experiment_id: str,
    control_id: str,
    control_n: int,
    control_conv: int,
    treatment_id: str,
    treatment_n: int,
    treatment_conv: int,
) -> ExperimentResults:
    return ExperimentResults(
        experiment_id=experiment_id,
        total_subjects=control_n + treatment_n,
        variant_stats={
            control_id: VariantStats(
                variant_id=control_id,
                assignments=control_n,
                conversions=control_conv,
                conversion_rate=control_conv / control_n if control_n else 0.0,
                total_value=float(control_conv),
                avg_value=1.0 if control_conv else 0.0,
            ),
            treatment_id: VariantStats(
                variant_id=treatment_id,
                assignments=treatment_n,
                conversions=treatment_conv,
                conversion_rate=treatment_conv / treatment_n if treatment_n else 0.0,
                total_value=float(treatment_conv),
                avg_value=1.0 if treatment_conv else 0.0,
            ),
        },
    )


# ---------------------------------------------------------------------------
# Normal distribution utilities
# ---------------------------------------------------------------------------


def test_normal_cdf_known_values() -> None:
    assert abs(_normal_cdf(0.0) - 0.5) < 1e-9
    assert abs(_normal_cdf(1.96) - 0.975002) < 1e-5
    assert abs(_normal_cdf(-1.96) - 0.024998) < 1e-5
    assert abs(_normal_cdf(3.0) - 0.9986501) < 1e-6


def test_norm_ppf_roundtrip() -> None:
    for p in [0.025, 0.1, 0.5, 0.9, 0.975]:
        z = _norm_ppf(p)
        recovered = _normal_cdf(z)
        assert abs(recovered - p) < 1e-7, f"roundtrip failed at p={p}"


def test_norm_ppf_known_values() -> None:
    assert abs(_norm_ppf(0.975) - 1.96) < 0.005
    assert abs(_norm_ppf(0.5) - 0.0) < 1e-6
    assert abs(_norm_ppf(0.025) - (-1.96)) < 0.005


def test_p_value_two_tailed_z0() -> None:
    """z=0 → p_value = 1.0."""
    assert abs(_p_value_two_tailed(0.0) - 1.0) < 1e-6


def test_p_value_two_tailed_large_z() -> None:
    """Very large z → p_value ≈ 0."""
    assert _p_value_two_tailed(10.0) < 1e-20


def test_z_alpha_95() -> None:
    assert abs(_z_alpha(0.95) - 1.96) < 0.005


def test_z_alpha_99() -> None:
    assert abs(_z_alpha(0.99) - 2.576) < 0.005


# ---------------------------------------------------------------------------
# Wilson confidence intervals
# ---------------------------------------------------------------------------


def test_wilson_ci_zero_n() -> None:
    lower, upper = wilson_ci(0, 0)
    assert lower == 0.0
    assert upper == 1.0


def test_wilson_ci_all_converted() -> None:
    lower, upper = wilson_ci(100, 100)
    assert upper <= 1.0
    assert lower > 0.9


def test_wilson_ci_none_converted() -> None:
    lower, upper = wilson_ci(0, 100)
    assert lower >= 0.0
    assert upper < 0.1


def test_wilson_ci_bounds_valid() -> None:
    for n in [10, 50, 100, 500]:
        for c in [0, 1, n // 4, n // 2, n]:
            lower, upper = wilson_ci(c, n)
            assert 0.0 <= lower <= upper <= 1.0


def test_wilson_ci_midpoint_close_to_p_hat() -> None:
    """For large n, the Wilson CI centre should be close to p_hat."""
    n = 1000
    c = 200
    lower, upper = wilson_ci(c, n)
    mid = (lower + upper) / 2
    p_hat = c / n
    assert abs(mid - p_hat) < 0.01


def test_wilson_ci_symmetric_around_0_5() -> None:
    lower, upper = wilson_ci(50, 100)
    assert abs((lower + upper) / 2 - 0.5) < 0.01


def test_wilson_ci_99_wider_than_95() -> None:
    lo95, hi95 = wilson_ci(50, 100, confidence_level=0.95)
    lo99, hi99 = wilson_ci(50, 100, confidence_level=0.99)
    assert (hi99 - lo99) > (hi95 - lo95)


# ---------------------------------------------------------------------------
# Z-test for proportions
# ---------------------------------------------------------------------------


def test_z_test_identical_rates() -> None:
    """Same conversion rates → z ≈ 0, p ≈ 1, not significant."""
    result = z_test_proportions(50, 100, 50, 100)
    assert abs(result.z_score) < 0.001
    assert result.p_value > 0.9
    assert not result.is_significant


def test_z_test_large_difference_significant() -> None:
    """Large sample, 10% vs 20% uplift → should be significant."""
    result = z_test_proportions(100, 1000, 200, 1000)
    assert result.is_significant
    assert result.p_value < 0.05
    assert result.relative_uplift > 0


def test_z_test_small_sample_not_significant() -> None:
    """Small sample even with apparent difference → not significant."""
    result = z_test_proportions(1, 10, 3, 10)
    assert not result.is_significant


def test_z_test_negative_uplift() -> None:
    """Treatment worse than control → negative relative uplift."""
    result = z_test_proportions(200, 1000, 100, 1000)
    assert result.relative_uplift < 0
    assert result.z_score < 0


def test_z_test_zero_control_n() -> None:
    result = z_test_proportions(0, 0, 50, 100)
    assert not result.is_significant
    assert result.p_value == 1.0


def test_z_test_zero_treatment_n() -> None:
    result = z_test_proportions(50, 100, 0, 0)
    assert not result.is_significant


def test_z_test_zero_conversions() -> None:
    """0/n vs 0/n → z=0, p=1."""
    result = z_test_proportions(0, 100, 0, 100)
    assert abs(result.z_score) < 1e-9
    assert not result.is_significant


def test_z_test_ci_coverage() -> None:
    """CI for each variant should contain the observed rate."""
    result = z_test_proportions(100, 1000, 150, 1000)
    p_ctrl = 100 / 1000
    p_trt = 150 / 1000
    assert result.control_ci_lower <= p_ctrl <= result.control_ci_upper
    assert result.treatment_ci_lower <= p_trt <= result.treatment_ci_upper


def test_z_test_confidence_level_affects_significance() -> None:
    """Same data: significant at 95% but may not be at 99.9%."""
    result_95 = z_test_proportions(100, 1000, 200, 1000, confidence_level=0.95)
    result_999 = z_test_proportions(100, 1000, 200, 1000, confidence_level=0.999)
    # 95% result is significant; 99.9% may or may not be, but p-values are the same
    assert result_95.p_value == result_999.p_value
    if result_999.is_significant:
        assert result_95.is_significant  # 95% must also be significant if 99.9% is


def test_z_test_control_id_and_treatment_id_empty_from_function() -> None:
    """z_test_proportions returns empty IDs — caller fills them in."""
    result = z_test_proportions(50, 100, 50, 100)
    assert result.control_id == ""
    assert result.treatment_id == ""


def test_z_test_model_copy_fills_ids() -> None:
    result = z_test_proportions(50, 100, 50, 100)
    result = result.model_copy(update={"control_id": "ctrl", "treatment_id": "trt"})
    assert result.control_id == "ctrl"
    assert result.treatment_id == "trt"


# ---------------------------------------------------------------------------
# Winner detection
# ---------------------------------------------------------------------------


def test_detect_winner_no_winner_small_sample() -> None:
    experiment = _make_experiment()
    results = _make_results("test-exp", "control", 10, 1, "treatment", 10, 3)
    winner = detect_winner(results, experiment)
    assert winner.winner_id is None
    assert len(winner.comparisons) == 1


def test_detect_winner_clear_winner() -> None:
    experiment = _make_experiment()
    results = _make_results("test-exp", "control", 1000, 100, "treatment", 1000, 200)
    winner = detect_winner(results, experiment)
    assert winner.winner_id == "treatment"


def test_detect_winner_treatment_worse() -> None:
    """Treatment significantly worse than control → no winner (need positive uplift)."""
    experiment = _make_experiment()
    results = _make_results("test-exp", "control", 1000, 200, "treatment", 1000, 100)
    winner = detect_winner(results, experiment)
    assert winner.winner_id is None


def test_detect_winner_single_variant() -> None:
    experiment = Experiment(
        id="solo",
        name="Solo",
        variants=[Variant(id="only", name="Only", weight=1.0)],
        start_date=datetime(2026, 1, 1, tzinfo=UTC),
    )
    results = ExperimentResults(
        experiment_id="solo",
        total_subjects=100,
        variant_stats={
            "only": VariantStats(
                variant_id="only",
                assignments=100,
                conversions=20,
                conversion_rate=0.2,
                total_value=20.0,
                avg_value=1.0,
            )
        },
    )
    winner = detect_winner(results, experiment)
    assert winner.winner_id is None
    assert winner.comparisons == []


def test_detect_winner_invalid_control() -> None:
    experiment = _make_experiment()
    results = _make_results("test-exp", "control", 1000, 100, "treatment", 1000, 200)
    with pytest.raises(ValueError, match="Control variant"):
        detect_winner(results, experiment, control_id="nonexistent")


def test_detect_winner_custom_control() -> None:
    """Use treatment as control — now control is the 'good' arm."""
    experiment = _make_experiment()
    results = _make_results("test-exp", "control", 1000, 100, "treatment", 1000, 200)
    # Use 'treatment' (200/1000) as control, 'control' (100/1000) as treatment → negative uplift
    winner = detect_winner(results, experiment, control_id="treatment")
    assert winner.winner_id is None  # negative uplift → no winner


def test_detect_winner_comparisons_have_ids() -> None:
    experiment = _make_experiment()
    results = _make_results("test-exp", "control", 1000, 100, "treatment", 1000, 200)
    winner = detect_winner(results, experiment)
    assert len(winner.comparisons) == 1
    cmp = winner.comparisons[0]
    assert cmp.control_id == "control"
    assert cmp.treatment_id == "treatment"


def test_detect_winner_confidence_level_propagated() -> None:
    experiment = _make_experiment()
    results = _make_results("test-exp", "control", 1000, 100, "treatment", 1000, 200)
    winner = detect_winner(results, experiment, confidence_level=0.99)
    assert winner.confidence_level == 0.99
    assert all(c.confidence_level == 0.99 for c in winner.comparisons)


def test_detect_winner_three_variants() -> None:
    """Three variants: treatment-b beats control significantly, treatment-a doesn't."""
    experiment = Experiment(
        id="multi",
        name="Multi",
        variants=[
            Variant(id="control", name="Control", weight=0.34),
            Variant(id="trt-a", name="Treatment A", weight=0.33),
            Variant(id="trt-b", name="Treatment B", weight=0.33),
        ],
        start_date=datetime(2026, 1, 1, tzinfo=UTC),
    )
    results = ExperimentResults(
        experiment_id="multi",
        total_subjects=3000,
        variant_stats={
            "control": VariantStats(
                variant_id="control", assignments=1000, conversions=100,
                conversion_rate=0.1, total_value=100.0, avg_value=1.0,
            ),
            "trt-a": VariantStats(
                variant_id="trt-a", assignments=1000, conversions=105,
                conversion_rate=0.105, total_value=105.0, avg_value=1.0,
            ),
            "trt-b": VariantStats(
                variant_id="trt-b", assignments=1000, conversions=200,
                conversion_rate=0.2, total_value=200.0, avg_value=1.0,
            ),
        },
    )
    winner = detect_winner(results, experiment)
    assert winner.winner_id == "trt-b"
    assert len(winner.comparisons) == 2


# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------


def test_build_report_returns_experiment_report() -> None:
    experiment = _make_experiment()
    results = _make_results("test-exp", "control", 1000, 100, "treatment", 1000, 200)
    rpt = build_report(results, experiment)
    assert isinstance(rpt, ExperimentReport)
    assert rpt.experiment_id == "test-exp"
    assert rpt.experiment_name == "Test Experiment"
    assert rpt.total_subjects == 2000
    assert isinstance(rpt.winner, WinnerResult)
    assert rpt.generated_at is not None


def test_build_report_winner_id_propagated() -> None:
    experiment = _make_experiment()
    results = _make_results("test-exp", "control", 1000, 100, "treatment", 1000, 200)
    rpt = build_report(results, experiment)
    assert rpt.winner.winner_id == "treatment"


def test_build_report_no_winner() -> None:
    experiment = _make_experiment()
    results = _make_results("test-exp", "control", 10, 1, "treatment", 10, 2)
    rpt = build_report(results, experiment)
    assert rpt.winner.winner_id is None


def test_build_report_variant_stats_present() -> None:
    experiment = _make_experiment()
    results = _make_results("test-exp", "control", 500, 50, "treatment", 500, 75)
    rpt = build_report(results, experiment)
    assert "control" in rpt.variant_stats
    assert "treatment" in rpt.variant_stats
