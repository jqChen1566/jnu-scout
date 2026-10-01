"""Characterization tests for ``jnuscout.uq.calibration`` (pure NumPy).

The calibration convention under test (2026-08-24 review, item C2) is

    theta = Quantile(uq | err > epsilon_crit, q)

i.e. a low quantile of the disagreement over the *dangerous* configurations.

Every numeric expectation below is derived independently of the production
code path: either from the hand-written reference ``_manual_quantile``
(Hyndman-Fan type 7, the NumPy default) or from closed-form values written
out in the comments.
"""
import math

import numpy as np
import pytest

from jnuscout.uq.calibration import (
    calibrate_threshold,
    coverage_stats,
    expected_calibration_error,
    spearman_rho,
)


# --------------------------------------------------------------------------
# independent reference implementations
# --------------------------------------------------------------------------
def _manual_quantile(values, q):
    """Hyndman-Fan type 7 quantile with linear interpolation, hand-written."""
    xs = sorted(float(v) for v in values)
    assert xs, "reference quantile needs a non-empty sample"
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[int(lo)]
    return xs[lo] + (pos - lo) * (xs[hi] - xs[lo])


def _average_ranks(values):
    """O(n^2) average ranks (0-based) -- independent of the production argsort."""
    out = []
    for v in values:
        less = sum(1 for x in values if x < v)
        equal = sum(1 for x in values if x == v)
        out.append(less + (equal - 1) / 2.0)
    return np.array(out, dtype=float)


# --------------------------------------------------------------------------
# calibrate_threshold
# --------------------------------------------------------------------------
def test_calibrate_threshold_matches_manual_quantile_of_dangerous_set():
    uq = np.array([0.10, 0.25, 0.30, 0.55, 0.60, 0.80, 0.95, 1.20])
    err = np.array([0.00, 0.01, 0.02, 0.05, 0.40, 0.90, 2.00, 5.00])
    # err > 0.1 selects uq values [0.60, 0.80, 0.95, 1.20]
    # q = 0.05 on n=4: pos = 0.15 -> 0.60 + 0.15 * (0.80 - 0.60) = 0.63
    theta = calibrate_threshold(uq, err, epsilon_crit=0.1)
    assert theta == pytest.approx(0.63, rel=1e-12)
    assert theta == pytest.approx(_manual_quantile([0.60, 0.80, 0.95, 1.20], 0.05), rel=1e-12)


def test_calibrate_threshold_ignores_safe_configurations():
    """Only uq values with err > epsilon_crit enter the quantile."""
    uq = np.array([0.10, 0.25, 0.30, 0.55, 0.60, 0.80, 0.95, 1.20])
    err = np.array([0.00, 0.01, 0.02, 0.05, 0.40, 0.90, 2.00, 5.00])
    theta_ref = calibrate_threshold(uq, err, epsilon_crit=0.1)

    uq_safe_inflated = uq.copy()
    uq_safe_inflated[:4] = 1.0e3  # the safe configurations get huge uq values
    assert calibrate_threshold(uq_safe_inflated, err, epsilon_crit=0.1) == pytest.approx(
        theta_ref, rel=1e-12
    )


def test_calibrate_threshold_dangerous_set_uses_strict_inequality():
    """err exactly equal to epsilon_crit is *not* dangerous."""
    uq = np.array([1.0, 2.0, 3.0])
    err = np.array([0.5, 0.5, 0.6])
    theta = calibrate_threshold(uq, err, epsilon_crit=0.5)
    # dangerous set is {3.0} only (err == 0.5 is excluded); including the
    # equality would have given Quantile([1,2,3], 0.05) = 1.1
    assert theta == pytest.approx(3.0, rel=1e-12)


def test_calibrate_threshold_empty_dangerous_set_returns_inf():
    uq = np.array([0.1, 0.5, 0.9])
    err = np.array([0.0, 0.2, 0.3])
    theta = calibrate_threshold(uq, err, epsilon_crit=0.5)
    assert math.isinf(theta) and theta > 0


def test_calibrate_threshold_single_dangerous_configuration():
    uq = np.array([0.1, 0.7, 0.3])
    err = np.array([0.0, 1.0, 0.0])
    for q in (0.0, 0.05, 0.5, 0.9, 1.0):
        assert calibrate_threshold(uq, err, epsilon_crit=0.5, quantile=q) == pytest.approx(
            0.7, rel=1e-12
        )


def test_calibrate_threshold_quantile_endpoints_and_monotonicity():
    uq = np.array([0.1, 0.4, 0.2, 0.9, 0.5, 1.3, 0.3])
    err = np.array([0.0, 1.2, 0.0, 2.0, 1.5, 0.0, 0.0])
    dangerous = [0.4, 0.9, 0.5]  # uq where err > 1.0 (entry 2 has err == 1.2)
    assert calibrate_threshold(uq, err, 1.0, quantile=0.0) == pytest.approx(min(dangerous), rel=1e-12)
    assert calibrate_threshold(uq, err, 1.0, quantile=1.0) == pytest.approx(max(dangerous), rel=1e-12)
    thetas = [calibrate_threshold(uq, err, 1.0, quantile=q) for q in (0.05, 0.5, 0.95)]
    assert thetas[0] <= thetas[1] <= thetas[2]


def test_calibrate_threshold_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        calibrate_threshold(np.zeros(4), np.zeros(3), epsilon_crit=0.1)
    with pytest.raises(ValueError, match="shape mismatch"):
        calibrate_threshold(np.zeros((2, 2)), np.zeros(4), epsilon_crit=0.1)


def test_calibrate_threshold_accepts_lists_and_returns_plain_float():
    theta = calibrate_threshold([0.1, 0.6, 0.8], [0.0, 1.0, 1.0], epsilon_crit=0.5, quantile=0.5)
    assert theta == pytest.approx(0.7, rel=1e-12)  # median of [0.6, 0.8]
    assert type(theta) is float  # documented return type is a Python float


# --------------------------------------------------------------------------
# spearman_rho
# --------------------------------------------------------------------------
def test_spearman_perfect_monotone_and_antitone():
    uq = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman_rho(uq, 3.0 * uq ** 3 + 1.0) == pytest.approx(1.0, abs=1e-12)
    assert spearman_rho(uq, -uq) == pytest.approx(-1.0, abs=1e-12)


def test_spearman_matches_independent_average_rank_pearson():
    uq = [0.5, 0.5, 1.5, 2.0, 2.0, 2.0, 3.5, 4.0, 4.5, 5.0]
    err = [1.0, 2.0, 2.0, 3.0, 5.0, 4.0, 6.0, 7.0, 6.0, 9.0]
    ru = _average_ranks(uq)
    re = _average_ranks(err)
    ref = float(np.corrcoef(ru, re)[0, 1])
    assert spearman_rho(uq, err) == pytest.approx(ref, rel=1e-12)


def test_spearman_ties_closed_form():
    # uq ranks = [0.5, 0.5, 2.5, 2.5, 4.5, 4.5], err ranks = [0..5]
    # centred dot = 5+3+0+0+3+5 = 16; norms^2 = 16 and 17.5
    expected = 16.0 / math.sqrt(16.0 * 17.5)
    assert spearman_rho([1, 1, 2, 2, 3, 3], [1, 2, 3, 4, 5, 6]) == pytest.approx(
        expected, rel=1e-12
    )
    assert expected == pytest.approx(16.0 / math.sqrt(280.0), rel=1e-12)


def test_spearman_small_samples_and_constant_input_return_nan():
    assert math.isnan(spearman_rho([], []))
    assert math.isnan(spearman_rho([1.0], [2.0]))
    assert math.isnan(spearman_rho([1.0, 2.0], [2.0, 1.0]))
    assert math.isnan(spearman_rho([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]))


def test_spearman_shape_mismatch_raises():
    """Same contract as calibrate_threshold (2026-10-01 review).

    Before the fix a length mismatch either raised an opaque broadcasting error
    or -- for err of length 1 -- silently returned nan.
    """
    with pytest.raises(ValueError, match="shape mismatch"):
        spearman_rho([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="shape mismatch"):
        spearman_rho(np.zeros((2, 2)), np.zeros(4))
    with pytest.raises(ValueError, match="shape mismatch"):
        spearman_rho([1.0, 2.0, 3.0], [1.0])  # previously a silent nan
    with pytest.raises(ValueError, match="shape mismatch"):
        spearman_rho([1.0, 2.0], [1.0, 2.0, 3.0])  # well below the size guard


# --------------------------------------------------------------------------
# expected_calibration_error
# --------------------------------------------------------------------------
def test_ece_hand_computed_half_dangerous():
    """20 samples, 10 bins of 2: the 5 lowest-uq bins are safe, the rest dangerous."""
    uq = np.arange(20, dtype=float)
    err = np.array([0.0] * 10 + [1.0] * 10)
    # nominal_i = 1 - (i + 0.5)/10 = 0.95, 0.85, ..., 0.05
    # |actual - nominal| sums to 3.75 in each half; weight per bin 2/20 = 0.1
    # ECE = 0.1 * (3.75 + 3.75) = 0.75
    assert expected_calibration_error(uq, err, epsilon_crit=0.5, n_bins=10) == pytest.approx(
        0.75, rel=1e-12
    )


def test_ece_all_safe_closed_form():
    # actual = 0 in every bin -> ECE is the mean nominal rate = 0.5
    uq = np.arange(20, dtype=float)
    err = np.zeros(20)
    assert expected_calibration_error(uq, err, epsilon_crit=0.5, n_bins=10) == pytest.approx(
        0.5, rel=1e-12
    )


def test_ece_requires_two_samples_per_bin():
    assert math.isnan(expected_calibration_error(np.zeros(0), np.zeros(0), 0.5, n_bins=10))
    assert math.isnan(expected_calibration_error(np.arange(19.0), np.zeros(19), 0.5, n_bins=10))
    # exactly n_bins * 2 samples is enough
    assert not math.isnan(expected_calibration_error(np.arange(20.0), np.zeros(20), 0.5, n_bins=10))


def test_ece_depends_only_on_the_uq_ordering():
    uq = np.arange(20, dtype=float)
    err = np.array([0.0] * 10 + [1.0] * 10)
    ref = expected_calibration_error(uq, err, epsilon_crit=0.5, n_bins=10)
    # rescaling uq cannot move samples between bins (binning is rank based)
    assert expected_calibration_error(uq * 1000.0 + 7.0, err, epsilon_crit=0.5, n_bins=10) == ref


def test_ece_is_order_sensitive_within_uq_ties():
    """Documents current behaviour: ties in uq are broken by input order.

    Two datasets describing the same *values* but with the two middle samples
    swapped give different ECEs (0.75 vs 0.65), because argsort(mergesort)
    keeps the input order inside a tie group.
    """
    uq = np.ones(20)
    err_a = np.array([0.0] * 10 + [1.0] * 10)
    err_b = err_a.copy()
    err_b[9], err_b[10] = err_b[10], err_b[9]
    ece_a = expected_calibration_error(uq, err_a, epsilon_crit=0.5, n_bins=10)
    ece_b = expected_calibration_error(uq, err_b, epsilon_crit=0.5, n_bins=10)
    assert ece_a == pytest.approx(0.75, rel=1e-12)
    # bins 4 and 5 become mixed (|0.5 - 0.55| and |0.5 - 0.45|):
    # 0.1 * (0.95+0.85+0.75+0.65 + 0.05+0.05 + 0.65+0.75+0.85+0.95) = 0.65
    assert ece_b == pytest.approx(0.65, rel=1e-12)


# --------------------------------------------------------------------------
# coverage_stats
# --------------------------------------------------------------------------
def test_coverage_stats_hand_computed():
    uq = np.array([0.1, 0.2, 0.3, 0.4])
    err = np.array([0.0, 5.0, 0.0, 5.0])
    stats = coverage_stats(uq, err, theta=0.25, epsilon_crit=1.0)
    assert set(stats) == {
        "n", "coverage", "fallback_rate", "n_dangerous",
        "false_negative_rate", "nn_mae", "all_mae", "theta",
    }
    assert stats["n"] == 4
    assert stats["coverage"] == pytest.approx(0.5, rel=1e-12)          # uq < 0.25: first two
    assert stats["fallback_rate"] == pytest.approx(0.5, rel=1e-12)
    assert stats["n_dangerous"] == 2                                    # err > 1: entries 2 and 4
    assert stats["false_negative_rate"] == pytest.approx(0.5, rel=1e-12)  # 1 missed of 2 dangerous
    assert stats["nn_mae"] == pytest.approx(2.5, rel=1e-12)             # mean(err[:2]) = (0+5)/2
    assert stats["all_mae"] == pytest.approx(2.5, rel=1e-12)
    assert stats["theta"] == pytest.approx(0.25, rel=1e-12)


def test_coverage_stats_empty_input_returns_n_only():
    assert coverage_stats([], [], theta=0.5, epsilon_crit=0.1) == {"n": 0}


def test_coverage_stats_uq_equal_to_theta_counts_as_fallback():
    """The NNP path is ``uq < theta`` -- equality falls back."""
    stats = coverage_stats([0.5, 0.5], [0.0, 0.0], theta=0.5, epsilon_crit=1.0)
    assert stats["coverage"] == 0.0
    assert stats["fallback_rate"] == 1.0


def test_coverage_stats_no_dangerous_configurations():
    stats = coverage_stats([0.1, 0.2], [0.0, 0.1], theta=0.5, epsilon_crit=1.0)
    assert stats["n_dangerous"] == 0
    assert stats["false_negative_rate"] == 0.0  # guarded branch, not 0/0


def test_coverage_stats_nn_mae_nan_when_everything_falls_back():
    stats = coverage_stats([0.9, 0.8], [1.0, 2.0], theta=0.5, epsilon_crit=0.5)
    assert stats["coverage"] == 0.0
    assert math.isnan(stats["nn_mae"])
    assert stats["all_mae"] == pytest.approx(1.5, rel=1e-12)


def test_calibration_pipeline_empty_dangerous_set_never_falls_back():
    """theta = inf from an empty dangerous set -> the whole set runs on the NNP."""
    uq = np.array([0.1, 0.4, 0.9])
    err = np.array([0.0, 0.2, 0.3])
    theta = calibrate_threshold(uq, err, epsilon_crit=0.5)
    assert math.isinf(theta)
    stats = coverage_stats(uq, err, theta, epsilon_crit=0.5)
    assert stats["coverage"] == 1.0
    assert stats["fallback_rate"] == 0.0
    assert stats["n_dangerous"] == 0
    assert stats["false_negative_rate"] == 0.0
    assert stats["nn_mae"] == pytest.approx(stats["all_mae"], rel=1e-12)
