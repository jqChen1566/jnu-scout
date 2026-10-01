"""Characterization tests for ``jnuscout.calculators.committee`` disagreement metrics.

Pure computation only: no model weights are loaded. ``NNPCommittee.__init__``
and ``disagreement()`` never touch a backend -- the calculators are built
lazily in ``_get_calc``, which these tests do not call.

Metric semantics under test (module docstring):
    uq_E = std_i(E_i) / N            [eV/atom]  (same backend family only)
    uq_F = max_j max_i ||F_i,j - F_mean_j||      [eV/A]
"""
import logging
import math

import numpy as np
import pytest

pytest.importorskip("ase")

from jnuscout.calculators.committee import (  # noqa: E402
    DEFAULT_MEMBERS,
    NNPCommittee,
    _default_members,
)

MACE3 = [("m1", "mace", "w1"), ("m2", "mace", "w2"), ("m3", "mace", "w3")]


def _pred(energy, forces):
    return {"energy": float(energy), "forces": np.asarray(forces, dtype=float)}


def _committee(members, **kwargs):
    return NNPCommittee(members=members, device="cpu", **kwargs)


def _zeros(n_atoms):
    return np.zeros((n_atoms, 3), dtype=float)


# --------------------------------------------------------------------------
# zero / max / norm semantics
# --------------------------------------------------------------------------
def test_identical_members_have_zero_disagreement():
    forces = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    preds = {name: _pred(-100.0, forces.copy()) for name, _, _ in MACE3}
    rep = _committee(MACE3).disagreement(preds)
    assert rep["uq_energy"] == 0.0
    # uq_force is a max over ||F_i - mean(F)||: identical iterates give exactly
    # zero deviation only for exactly representable forces. With 0.1/0.2/0.3 the
    # mean carries a round-off of ~5.6e-17, so the documented behaviour is
    # "zero up to one ulp", not "bit-exactly zero".
    assert abs(rep["uq_force"]) <= 1e-15
    assert rep["energy_spread"] == 0.0
    assert rep["energy_spread_raw"] == 0.0
    assert rep["baselines_aligned"] is False
    assert rep["energy_mean"] == -100.0
    # the mean of three identical 0.1-style floats carries the same ulp round-off
    np.testing.assert_allclose(rep["forces_mean"], forces, rtol=1e-15)


def test_uq_force_is_bit_exactly_zero_for_dyadic_forces():
    """With exactly representable forces the identical-member case is exact 0."""
    forces = np.array([[0.5, 0.25, 0.75], [0.125, 0.0625, 0.3125]])
    preds = {name: _pred(-100.0, forces.copy()) for name, _, _ in MACE3}
    rep = _committee(MACE3).disagreement(preds)
    assert rep["uq_force"] == 0.0  # (x + x + x) / 3 == x exactly for dyadic x


def test_uq_force_is_per_atom_max_not_averaged_over_atoms():
    zeros = _zeros(3)
    deviating = np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    preds = {"m1": _pred(0.0, zeros), "m2": _pred(0.0, deviating), "m3": _pred(0.0, zeros.copy())}
    rep = _committee(MACE3).disagreement(preds)
    # mean force on atom 0 = (0 + 3 + 0)/3 = 1 -> deviations 1, 2, 1 -> max = 2
    assert rep["uq_force"] == pytest.approx(2.0, rel=1e-12)
    # averaging the per-atom deviation would dilute the single-atom anomaly to 2/3
    mean_over_atoms = float(np.linalg.norm(deviating - rep["forces_mean"], axis=1).mean())
    assert mean_over_atoms == pytest.approx(2.0 / 3.0, rel=1e-12)
    assert rep["uq_force"] == pytest.approx(3.0 * mean_over_atoms, rel=1e-12)


def test_uq_force_is_a_vector_norm_on_a_single_atom():
    members = [("a", "mace", "x"), ("b", "mace", "y")]
    preds = {
        "a": _pred(0.0, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
        "b": _pred(0.0, np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])),
    }
    rep = _committee(members).disagreement(preds)
    # |[3,4,0]| = 5; with two members each deviation from the mean is half of it -> 2.5
    assert rep["uq_force"] == pytest.approx(2.5, rel=1e-12)


def test_uq_force_matches_independent_loop_reference():
    forces = np.array([
        [[0.10, -0.20, 0.30], [0.00, 0.50, -0.10], [1.20, 0.00, -0.40], [0.20, 0.20, 0.20]],
        [[0.14, -0.18, 0.25], [0.10, 0.40, -0.10], [0.90, 0.30, -0.40], [0.20, 0.30, 0.10]],
        [[0.00, -0.30, 0.50], [-0.20, 0.60, 0.00], [1.50, -0.20, -0.50], [0.30, 0.10, 0.30]],
    ])
    names = ["a", "b", "c"]
    preds = {name: _pred(0.0, forces[k].copy()) for k, name in enumerate(names)}
    rep = _committee([(n, "mace", n) for n in names]).disagreement(preds)

    # independent reference: explicit loops, sqrt of the summed squares
    best = 0.0
    for j in range(forces.shape[1]):
        mean_j = forces[:, j].sum(axis=0) / forces.shape[0]
        for k in range(forces.shape[0]):
            d = forces[k, j] - mean_j
            best = max(best, math.sqrt(float((d * d).sum())))
    assert rep["uq_force"] == pytest.approx(best, rel=1e-12)
    np.testing.assert_allclose(rep["forces_mean"], forces.mean(axis=0), rtol=1e-12)


# --------------------------------------------------------------------------
# energy disagreement: population std over members, normalised by atom count
# --------------------------------------------------------------------------
def test_uq_energy_is_population_std_over_members_per_atom():
    n_atoms = 4
    preds = {name: _pred(e, _zeros(n_atoms)) for name, e in zip(["m1", "m2", "m3"], [0.0, 1.0, 2.0])}
    rep = _committee(MACE3).disagreement(preds)
    # population std (ddof=0) of [0, 1, 2] is sqrt(2/3)
    assert rep["uq_energy"] == pytest.approx(math.sqrt(2.0 / 3.0) / n_atoms, rel=1e-12)


def test_uq_energy_normalises_by_atom_count():
    """Two datasets with the same energy distribution but different N."""
    small = {name: _pred(e, _zeros(2)) for name, e in zip(["m1", "m2", "m3"], [0.0, 1.0, 2.0])}
    large = {name: _pred(e, _zeros(5)) for name, e in zip(["m1", "m2", "m3"], [0.0, 1.0, 2.0])}
    committee = _committee(MACE3)
    rep_small = committee.disagreement(small)
    rep_large = committee.disagreement(large)
    assert rep_small["uq_energy"] == pytest.approx(math.sqrt(2.0 / 3.0) / 2.0, rel=1e-12)
    assert rep_large["uq_energy"] == pytest.approx(math.sqrt(2.0 / 3.0) / 5.0, rel=1e-12)
    assert rep_small["uq_energy"] / rep_large["uq_energy"] == pytest.approx(2.5, rel=1e-12)


# --------------------------------------------------------------------------
# backend-family filtering of uq_energy
# --------------------------------------------------------------------------
def _mixed_family_case():
    members = [("mace-a", "mace", "x"), ("mace-b", "mace", "y"), ("aim", "aimnet2", "aimnet2_b973c")]
    preds = {
        "mace-a": _pred(0.0, _zeros(2)),
        "mace-b": _pred(2.0, _zeros(2)),
        "aim": _pred(1.0e6, _zeros(2)),   # absolute-energy reference of a different family
    }
    return members, preds


def test_uq_energy_ignores_other_backend_families():
    members, preds = _mixed_family_case()
    rep = _committee(members).disagreement(preds)
    # default energy_family="mace" -> std([0, 2]) = 1.0, then / N
    assert rep["uq_energy"] == pytest.approx(1.0 / 2.0, rel=1e-12)
    # identical forces everywhere -> no force disagreement
    assert rep["uq_force"] == 0.0
    # the raw/inter-family spread is still reported (diagnostic only)
    assert rep["energy_spread_raw"] == pytest.approx(1.0e6, rel=1e-12)
    assert rep["per_member_energy"]["aim"] == 1.0e6


def test_energy_family_selection():
    members, preds = _mixed_family_case()
    # an explicit single-member family gives std over one value -> 0
    assert _committee(members, energy_family="aimnet2").disagreement(preds)["uq_energy"] == 0.0
    # unknown/None family falls back to the largest family (mace, 2 members)
    for family in (None, "does-not-exist"):
        rep = _committee(members, energy_family=family).disagreement(preds)
        assert rep["uq_energy"] == pytest.approx(1.0 / 2.0, rel=1e-12)


def test_uq_force_spans_all_families():
    """Unlike uq_energy, the force metric is not family filtered."""
    members = [("mac", "mace", "x"), ("aim", "aimnet2", "aimnet2_b973c")]
    preds = {
        "mac": _pred(0.0, np.array([[0.0, 0.0, 0.0]])),
        "aim": _pred(0.0, np.array([[0.5, 0.0, 0.0]])),
    }
    rep = _committee(members).disagreement(preds)
    assert rep["uq_force"] == pytest.approx(0.25, rel=1e-12)  # half the difference
    assert rep["uq_energy"] == 0.0  # single mace member -> std over one value


# --------------------------------------------------------------------------
# baseline alignment (per-atom offsets)
# --------------------------------------------------------------------------
def test_baseline_alignment_cancels_a_per_atom_offset():
    members = [("A", "mace", "x"), ("B", "mace", "y")]
    n_atoms = 2
    # energies differ by (b_B - b_A) * N = 2 * 2 = 4 -> alignment makes them equal
    preds = {"A": _pred(10.0, _zeros(n_atoms)), "B": _pred(14.0, _zeros(n_atoms))}
    rep = _committee(members, energy_baselines={"A": 1.0, "B": 3.0}).disagreement(preds)
    assert rep["baselines_aligned"] is True
    assert rep["uq_energy"] == 0.0
    assert rep["energy_spread"] == 0.0            # aligned
    assert rep["energy_spread_raw"] == pytest.approx(4.0, rel=1e-12)
    # ensemble energy = aligned mean + the reference member's offset back
    assert rep["energy_mean"] == pytest.approx(8.0 + 1.0 * n_atoms, rel=1e-12)


def test_baseline_offset_is_scaled_by_atom_count():
    members = [("A", "mace", "x"), ("B", "mace", "y")]
    # fixed per-atom baselines at N=3: aligned = [10 - 3, 14 - 9] = [7, 5]
    preds = {"A": _pred(10.0, _zeros(3)), "B": _pred(14.0, _zeros(3))}
    rep = _committee(members, energy_baselines={"A": 1.0, "B": 3.0}).disagreement(preds)
    assert rep["uq_energy"] == pytest.approx(np.std([7.0, 5.0]) / 3.0, rel=1e-12)
    assert rep["uq_energy"] == pytest.approx(1.0 / 3.0, rel=1e-12)


# --------------------------------------------------------------------------
# contract / report structure
# --------------------------------------------------------------------------
def test_disagreement_requires_at_least_two_members():
    committee = _committee(MACE3)
    with pytest.raises(ValueError, match="at least 2 members"):
        committee.disagreement({})
    with pytest.raises(ValueError, match="at least 2 members"):
        committee.disagreement({"m1": _pred(0.0, _zeros(3))})


def test_report_keys_are_stable():
    preds = {name: _pred(0.0, _zeros(3)) for name, _, _ in MACE3}
    rep = _committee(MACE3).disagreement(preds)
    assert set(rep) == {
        "uq_energy", "uq_force", "energy_mean", "forces_mean",
        "energy_spread", "energy_spread_raw", "baselines_aligned", "per_member_energy",
    }
    assert set(rep["per_member_energy"]) == {"m1", "m2", "m3"}


# --------------------------------------------------------------------------
# warnings on degenerate inputs (2026-10-01 review): the computed values are
# deliberately left unchanged, only a log record is added
# --------------------------------------------------------------------------
_LOGGER = "jnuscout.calculators.committee"


def _committee_records(caplog):
    return [r for r in caplog.records if r.name == _LOGGER]


def test_mismatched_prediction_keys_warn_and_keep_the_degraded_value(caplog):
    preds = {"x": _pred(0.0, _zeros(2)), "y": _pred(-2078.0, _zeros(2))}
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        rep = _committee(MACE3).disagreement(preds)
    assert "do not match the committee members" in caplog.text
    assert "no member matched the predictions" in caplog.text
    # the degraded cross-family value is unchanged by the warning
    assert rep["uq_energy"] == pytest.approx(519.5, rel=1e-12)  # std([0, -2078]) / 2
    assert rep["energy_spread_raw"] == pytest.approx(2078.0, rel=1e-12)


def test_missing_member_predictions_warn_and_keep_values(caplog):
    preds = {"m1": _pred(0.0, _zeros(2)), "m2": _pred(2.0, _zeros(2))}
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        rep = _committee(MACE3).disagreement(preds)
    assert "members without predictions" in caplog.text
    assert "m3" in caplog.text
    assert rep["uq_energy"] == pytest.approx(0.5, rel=1e-12)  # std([0, 2]) / 2


def test_unavailable_energy_family_warns_and_falls_back(caplog):
    members, preds = _mixed_family_case()
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        rep = _committee(members, energy_family="xtb").disagreement(preds)
    assert "has no members among the predictions" in caplog.text
    # largest family (mace, two members) is still used -> value unchanged
    assert rep["uq_energy"] == pytest.approx(1.0 / 2.0, rel=1e-12)


def test_partial_energy_baselines_warn_and_keep_values(caplog):
    members = [("A", "mace", "x"), ("B", "mace", "y")]
    preds = {"A": _pred(10.0, _zeros(2)), "B": _pred(16.0, _zeros(2))}
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        rep = _committee(members, energy_baselines={"A": 1.0}).disagreement(preds)
    assert "no energy baseline" in caplog.text
    assert "B" in caplog.text
    # aligned = [10 - 1*2, 16 - 0] = [8, 16] -> std 4, / N=2 -> 2.0
    assert rep["uq_energy"] == pytest.approx(2.0, rel=1e-12)


def test_well_formed_inputs_emit_no_warning(caplog):
    members = [("A", "mace", "x"), ("B", "mace", "y")]
    preds = {"A": _pred(10.0, _zeros(2)), "B": _pred(16.0, _zeros(2))}
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _committee(members, energy_baselines={"A": 1.0, "B": 3.0}).disagreement(preds)
    assert _committee_records(caplog) == []


# --------------------------------------------------------------------------
# default member table (offline weight paths)
# --------------------------------------------------------------------------
def test_default_members_structure_and_backends():
    names = [m[0] for m in DEFAULT_MEMBERS]
    backends = [m[1] for m in DEFAULT_MEMBERS]
    assert names == ["MACE-MP-0", "MACE-MPA-0", "AIMNet2-b973c"]
    assert backends == ["mace", "mace", "aimnet2"]
    assert DEFAULT_MEMBERS[2][2] == "aimnet2_b973c"


def test_default_members_honour_environment_overrides(monkeypatch):
    monkeypatch.setenv("MACE_MP0_PATH", "local-mp0.model")
    monkeypatch.setenv("MACE_MPA0_PATH", "local-mpa0.model")
    members = _default_members()
    assert members[0] == ("MACE-MP-0", "mace", "local-mp0.model")
    assert members[1] == ("MACE-MPA-0", "mace", "local-mpa0.model")

    monkeypatch.delenv("MACE_MP0_PATH", raising=False)
    monkeypatch.delenv("MACE_MPA0_PATH", raising=False)
    members = _default_members()
    assert members[0][2] == "medium"  # resolved online by mace_mp
    assert members[1][2].endswith("mace-mpa-0-medium.model")
