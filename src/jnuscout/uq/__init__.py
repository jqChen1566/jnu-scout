"""Uncertainty quantification: committee disagreement metrics and threshold
calibration (Phase 2).

Calibration formula (corrected in the 2026-08-24 review, item C2):
    theta = Quantile(uq | err > epsilon_crit, 5%)
i.e. the low quantile of the uq distribution over the "dangerous"
configurations (true error above the threshold) -- it guarantees that 95% of
the dangerous configurations have uq > theta, so the fallback threshold theta
captures the vast majority of high-error configurations (false negatives < 5%).

The earlier statement "taking the 95th percentile of uq -> guaranteed false
negatives < 5%" is wrong: a high quantile of uq only describes the tail of the
uq distribution itself and has no logical relation to the fraction of
high-error configurations that go undetected.
"""
from jnuscout.uq.calibration import (
    calibrate_threshold,
    spearman_rho,
    expected_calibration_error,
    coverage_stats,
)

__all__ = [
    "calibrate_threshold",
    "spearman_rho",
    "expected_calibration_error",
    "coverage_stats",
]
