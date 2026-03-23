"""Statistical analysis for A/B experiments.

Implements:
- Wilson confidence intervals for proportions
- Z-test for two proportions (one-sided and two-tailed)
- Winner detection with configurable significance threshold
- Full experiment report generation

No scipy required — uses Python's built-in math module.
"""

import math
from datetime import UTC, datetime

from rc_experiment_runner.models import (
    Experiment,
    ExperimentReport,
    ExperimentResults,
    StatisticalResult,
    WinnerResult,
)


def _normal_cdf(z: float) -> float:
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _p_value_two_tailed(z: float) -> float:
    """Two-tailed p-value for a z-score."""
    return 2.0 * (1.0 - _normal_cdf(abs(z)))


def _z_alpha(confidence_level: float) -> float:
    """Critical z value for a given confidence level (two-tailed)."""
    # We need z such that Φ(z) = (1 + confidence_level) / 2
    # Approximate via Newton's method on the normal CDF.
    alpha = 1.0 - confidence_level
    p_target = 1.0 - alpha / 2.0  # upper tail quantile
    return _norm_ppf(p_target)


def _norm_ppf(p: float) -> float:
    """Percent point function (inverse CDF) of the standard normal distribution.

    Uses the rational approximation from Peter Acklam's algorithm.
    Accurate to about 1e-9.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    # Coefficients for the rational approximation
    a = [-3.969683028665376e01, 2.209460984245205e02,
         -2.759285104469687e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02,
         -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e00, 3.754408661907416e00]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )


def wilson_ci(conversions: int, n: int, confidence_level: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    The Wilson interval has better coverage properties than the normal
    approximation (Wald interval), especially for small samples or
    extreme proportions.

    Returns (lower, upper) bounds.
    """
    if n == 0:
        return (0.0, 1.0)

    z = _z_alpha(confidence_level)
    z2 = z * z
    p_hat = conversions / n

    center = (p_hat + z2 / (2 * n)) / (1 + z2 / n)
    margin = (z / (1 + z2 / n)) * math.sqrt(
        p_hat * (1 - p_hat) / n + z2 / (4 * n * n)
    )

    return (max(0.0, center - margin), min(1.0, center + margin))


def z_test_proportions(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    confidence_level: float = 0.95,
) -> StatisticalResult:
    """Two-proportion z-test comparing treatment vs. control.

    Uses pooled proportion under the null hypothesis (H0: p1 == p2).
    Returns a StatisticalResult with z-score, p-value, CIs, and significance flag.
    """
    p_control = control_conversions / control_n if control_n > 0 else 0.0
    p_treatment = treatment_conversions / treatment_n if treatment_n > 0 else 0.0

    # Wilson CIs for each variant individually
    control_ci = wilson_ci(control_conversions, control_n, confidence_level)
    treatment_ci = wilson_ci(treatment_conversions, treatment_n, confidence_level)

    # Relative uplift
    relative_uplift = (
        (p_treatment - p_control) / p_control if p_control > 0 else float("inf")
    )

    # Handle degenerate cases
    if control_n == 0 or treatment_n == 0:
        return StatisticalResult(
            control_id="",  # filled in by caller
            treatment_id="",
            control_rate=p_control,
            treatment_rate=p_treatment,
            control_ci_lower=control_ci[0],
            control_ci_upper=control_ci[1],
            treatment_ci_lower=treatment_ci[0],
            treatment_ci_upper=treatment_ci[1],
            relative_uplift=relative_uplift,
            z_score=0.0,
            p_value=1.0,
            confidence_level=confidence_level,
            is_significant=False,
        )

    # Pooled proportion
    p_pool = (control_conversions + treatment_conversions) / (control_n + treatment_n)

    if p_pool <= 0.0 or p_pool >= 1.0:
        # No variance — no test possible
        z_score = 0.0
        p_value = 1.0
    else:
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / control_n + 1 / treatment_n))
        z_score = (p_treatment - p_control) / se if se > 0 else 0.0
        p_value = _p_value_two_tailed(z_score)

    alpha = 1.0 - confidence_level
    is_significant = p_value < alpha

    return StatisticalResult(
        control_id="",  # filled in by caller
        treatment_id="",
        control_rate=p_control,
        treatment_rate=p_treatment,
        control_ci_lower=control_ci[0],
        control_ci_upper=control_ci[1],
        treatment_ci_lower=treatment_ci[0],
        treatment_ci_upper=treatment_ci[1],
        relative_uplift=relative_uplift,
        z_score=z_score,
        p_value=p_value,
        confidence_level=confidence_level,
        is_significant=is_significant,
    )


def detect_winner(
    results: ExperimentResults,
    experiment: Experiment,
    control_id: str | None = None,
    confidence_level: float = 0.95,
) -> WinnerResult:
    """Detect the winning variant in an experiment.

    Compares each non-control variant against the control using a two-proportion
    z-test. A variant is declared a winner if it achieves statistical significance
    AND has a positive uplift.

    If control_id is not specified, uses the first variant as control.
    If multiple variants beat the control, the one with the highest conversion
    rate is declared the winner.

    Returns a WinnerResult with all pairwise comparisons.
    """
    variant_ids = list(results.variant_stats.keys())
    if not variant_ids:
        return WinnerResult(
            experiment_id=results.experiment_id,
            winner_id=None,
            confidence_level=confidence_level,
            comparisons=[],
        )

    # Resolve control variant
    if control_id is None:
        control_id = experiment.variants[0].id if experiment.variants else variant_ids[0]

    if control_id not in results.variant_stats:
        raise ValueError(f"Control variant '{control_id}' not in results")

    control_stats = results.variant_stats[control_id]
    comparisons: list[StatisticalResult] = []
    candidates: list[tuple[float, str]] = []  # (conversion_rate, variant_id)

    for vid, stats in results.variant_stats.items():
        if vid == control_id:
            continue

        result = z_test_proportions(
            control_conversions=control_stats.conversions,
            control_n=control_stats.assignments,
            treatment_conversions=stats.conversions,
            treatment_n=stats.assignments,
            confidence_level=confidence_level,
        )
        result = result.model_copy(update={"control_id": control_id, "treatment_id": vid})
        comparisons.append(result)

        if result.is_significant and result.relative_uplift > 0:
            candidates.append((stats.conversion_rate, vid))

    winner_id: str | None = None
    if candidates:
        # Pick the variant with the highest conversion rate among significant winners
        candidates.sort(reverse=True)
        winner_id = candidates[0][1]

    return WinnerResult(
        experiment_id=results.experiment_id,
        winner_id=winner_id,
        confidence_level=confidence_level,
        comparisons=comparisons,
    )


def build_report(
    results: ExperimentResults,
    experiment: Experiment,
    control_id: str | None = None,
    confidence_level: float = 0.95,
) -> ExperimentReport:
    """Build a full analysis report for an experiment."""
    winner = detect_winner(results, experiment, control_id, confidence_level)
    return ExperimentReport(
        experiment_id=results.experiment_id,
        experiment_name=experiment.name,
        total_subjects=results.total_subjects,
        variant_stats=results.variant_stats,
        winner=winner,
        generated_at=datetime.now(UTC),
    )
