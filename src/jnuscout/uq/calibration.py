"""Uncertainty-quantification threshold calibration and calibration metrics
(Phase 2).

All formulas follow the convention corrected in the 2026-08-24 review (item C2).

Symbols
-------
uq : array (n,)         committee disagreement score per configuration (the
                        Safe-NNP uncertainty estimate)
err : array (n,)        true error per configuration (against the reference
                        calculation, e.g. per-atom force error in eV/Angstrom)
epsilon_crit : float    danger threshold -- a configuration whose error
                        exceeds it counts as "dangerous"
"""
from typing import Tuple

import numpy as np


def calibrate_threshold(uq, err, epsilon_crit, quantile: float = 0.05) -> float:
    """Calibrate the fallback threshold ``theta = Quantile(uq | err > epsilon_crit, 5%)``.

    Meaning: take the low quantile of ``uq`` only over the "dangerous"
    configurations (true error > ``epsilon_crit``); consequently, among the
    configurations below ``theta`` the fraction of dangerous ones is <= 5%
    (false negatives are controlled).

    Contrast with the incorrect recipe "the 95th percentile of uq": that
    describes the upper tail of the ``uq`` distribution itself, which may fall
    between two modes and gives no guarantee that dangerous configurations are
    captured.

    Parameters
    ----------
    uq, err : array-like
    epsilon_crit : float
    quantile : float
        Quantile of the dangerous set (0.05 -> covers 95% of the dangerous
        configurations).

    Returns
    -------
    float
        theta; if the dangerous set is empty, ``np.inf`` is returned (never
        fall back; the caller must raise an alarm).
    """
    uq = np.asarray(uq, dtype=float)
    err = np.asarray(err, dtype=float)
    if uq.shape != err.shape:
        raise ValueError(f"shape mismatch: uq {uq.shape} vs err {err.shape}")
    dangerous = uq[err > epsilon_crit]
    if dangerous.size == 0:
        return float("inf")
    return float(np.quantile(dangerous, quantile))


def spearman_rho(uq, err) -> float:
    """Spearman rank correlation (validates "disagreement correlates with
    error"; the roadmap requires rho > 0.6).

    Implemented as the Pearson correlation of ranks (no scipy dependency).
    """
    uq = np.asarray(uq, dtype=float)
    err = np.asarray(err, dtype=float)
    if uq.size < 3:
        return float("nan")

    def _rank(a):
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(a.size, dtype=float)
        # Average ranks to handle ties
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.zeros(counts.size)
        np.add.at(sums, inv, ranks)
        return (sums / counts)[inv]

    ru, re = _rank(uq), _rank(err)
    ru = ru - ru.mean()
    re = re - re.mean()
    denom = np.sqrt((ru ** 2).sum() * (re ** 2).sum())
    return float((ru * re).sum() / denom) if denom > 0 else float("nan")


def expected_calibration_error(uq, err, epsilon_crit, n_bins: int = 10) -> float:
    """ECE: over uq-sorted bins, the weighted mean of |actual danger rate -
    nominal danger rate|.

    nominal danger rate = 1 - uq order quantile of the configurations in the
    bin (approximated by the bin-centre quantile);
    actual danger rate = fraction of configurations with err > epsilon_crit in
    the bin.

    Note: this implementation uses the simplified "uq-quantile binning"
    convention -- the nominal risk is given by the bin position, unlike
    classification-style ECE (predicted probability vs accuracy); it is meant
    only for cross-method comparison.
    """
    uq = np.asarray(uq, dtype=float)
    err = np.asarray(err, dtype=float)
    n = uq.size
    if n < n_bins * 2:
        return float("nan")
    order = np.argsort(uq, kind="mergesort")
    bins = np.array_split(order, n_bins)
    ece = 0.0
    for i, idx in enumerate(bins):
        if idx.size == 0:
            continue
        actual = float(np.mean(err[idx] > epsilon_crit))
        # Nominal: the uq tail probability at the bin centre (further right = "more dangerous")
        nominal = 1.0 - (i + 0.5) / n_bins
        ece += (idx.size / n) * abs(actual - nominal)
    return float(ece)


def coverage_stats(uq, err, theta, epsilon_crit) -> dict:
    """Coverage / fallback / missed-detection statistics (roadmap acceptance
    criteria).

    coverage = fraction with uq < theta (the fraction handled by the NNP --
               the name follows the roadmap; its actual meaning is "share
               carried by the NNP", and fallback_rate = 1 - coverage)
    false_negative_rate = fraction with uq < theta and err > epsilon_crit
               (share of the dangerous configurations that were not sent back)
    """
    uq = np.asarray(uq, dtype=float)
    err = np.asarray(err, dtype=float)
    n = uq.size
    if n == 0:
        return {"n": 0}
    nn_path = uq < theta
    dangerous = err > epsilon_crit
    missed = nn_path & dangerous
    n_dangerous = int(dangerous.sum())
    return {
        "n": int(n),
        "coverage": float(nn_path.mean()),
        "fallback_rate": float(1.0 - nn_path.mean()),
        "n_dangerous": n_dangerous,
        "false_negative_rate": float(missed.sum() / n_dangerous) if n_dangerous else 0.0,
        "nn_mae": float(err[nn_path].mean()) if nn_path.any() else float("nan"),
        "all_mae": float(err.mean()),
        "theta": float(theta),
    }
