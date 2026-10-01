"""Characterization tests for ``SafeNNPCalculator`` (UQ-triggered fallback).

No real model is loaded: a stub committee returns a fixed report and a stub
fallback calculator records its invocations. The production ``ORCACalculator``
symbol inside ``safe_nnp`` is monkeypatched so that a regression in the
decision rule cannot launch a real ORCA process.
"""
import numpy as np
import pytest

pytest.importorskip("ase")

from ase import Atoms  # noqa: E402
from ase.calculators.calculator import Calculator, all_changes  # noqa: E402

import jnuscout.calculators.safe_nnp as safe_nnp_module  # noqa: E402
from jnuscout.calculators.committee import NNPCommittee  # noqa: E402
from jnuscout.calculators.safe_nnp import SafeNNPCalculator  # noqa: E402

N_ATOMS = 3

FALLBACK_ENERGY = -9.75
FALLBACK_FORCE = 0.25


class StubCommittee:
    """Returns a fixed disagreement report; records calls and the atoms it saw."""

    def __init__(self, uq_force, uq_energy, energy=-1.5):
        self.uq_force = float(uq_force)
        self.uq_energy = float(uq_energy)
        self.energy = float(energy)
        self.forces = np.full((N_ATOMS, 3), 0.125)
        self.calls = 0
        self.seen_atoms = None
        self.last_report = None

    def compute(self, atoms):
        self.calls += 1
        self.seen_atoms = atoms
        self.last_report = {
            "uq_energy": self.uq_energy,
            "uq_force": self.uq_force,
            "energy_mean": self.energy,
            "forces_mean": self.forces,
        }
        return self.last_report


class StubFallback(Calculator):
    """Minimal ASE calculator standing in for ORCACalculator."""

    implemented_properties = ["energy", "forces"]
    instances = []

    def __init__(self, level=None, **kwargs):
        super().__init__()
        self.level = level
        self.kwargs = dict(kwargs)
        self.calls = 0
        StubFallback.instances.append(self)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        self.results = {
            "energy": FALLBACK_ENERGY,
            "forces": np.full((len(atoms), 3), FALLBACK_FORCE),
        }


@pytest.fixture
def stub_fallback(monkeypatch):
    StubFallback.instances.clear()
    monkeypatch.setattr(safe_nnp_module, "ORCACalculator", StubFallback)
    yield StubFallback
    StubFallback.instances.clear()


def _atoms(offset=0.0):
    return Atoms(
        "H2O",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, offset]],
    )


def _make(uq_force, uq_energy, theta_force=0.05, theta_energy=None, use_or_rule=True, **kwargs):
    committee = StubCommittee(uq_force, uq_energy)
    calc = SafeNNPCalculator(
        theta_force=theta_force, theta_energy=theta_energy, committee=committee,
        device="cpu", use_or_rule=use_or_rule, **kwargs
    )
    return calc, committee


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------
def test_default_committee_and_lazy_fallback():
    calc = SafeNNPCalculator(theta_force=0.05)
    assert isinstance(calc.committee, NNPCommittee)
    assert calc._fallback is None  # no ORCA instance before the first fallback
    assert calc.n_calls == 0 and calc.n_fallback == 0
    assert calc.last_uq is None
    assert calc.stats() == {"n_calls": 0, "n_fallback": 0, "fallback_rate": 0.0}


# --------------------------------------------------------------------------
# decision rule
# --------------------------------------------------------------------------
def test_uq_below_theta_uses_the_committee_path(stub_fallback):
    calc, committee = _make(uq_force=0.01, uq_energy=1e-9)
    atoms = _atoms()
    atoms.calc = calc

    assert atoms.get_potential_energy() == committee.energy
    np.testing.assert_array_equal(atoms.get_forces(), committee.forces)

    assert calc.n_calls == 1
    assert calc.n_fallback == 0
    assert calc.last_path == "nnp"
    assert calc.last_uq is committee.last_report      # stored by reference
    assert committee.calls == 1                        # energy+forces share one call
    assert committee.seen_atoms is atoms               # the raw Atoms are forwarded
    assert stub_fallback.instances == []               # ORCA never constructed


def test_uq_equal_to_theta_triggers_fallback(stub_fallback):
    """Boundary: the rule is ``uq >= theta`` -> fall back."""
    calc, committee = _make(uq_force=0.05, uq_energy=0.0, theta_force=0.05)
    atoms = _atoms()
    atoms.calc = calc

    assert atoms.get_potential_energy() == FALLBACK_ENERGY
    np.testing.assert_array_equal(atoms.get_forces(), np.full((N_ATOMS, 3), FALLBACK_FORCE))
    assert calc.n_fallback == 1
    assert calc.last_path == "fallback"
    assert calc.last_uq["uq_force"] == 0.05
    assert len(stub_fallback.instances) == 1
    assert committee.calls == 1


def test_uq_above_theta_result_comes_from_the_fallback(stub_fallback):
    calc, committee = _make(
        uq_force=0.5, uq_energy=0.0, fallback_level="L3", fallback_kwargs={"nprocs": 4}
    )
    atoms = _atoms()
    atoms.calc = calc

    assert atoms.get_potential_energy() == FALLBACK_ENERGY
    np.testing.assert_array_equal(atoms.get_forces(), np.full((N_ATOMS, 3), FALLBACK_FORCE))
    assert calc.n_fallback == 1

    fb = stub_fallback.instances[0]
    assert fb.level == "L3"                 # fallback_level forwarded
    assert fb.kwargs == {"nprocs": 4}       # fallback_kwargs forwarded
    assert fb.calls == 1


def test_fallback_is_built_once_and_reused(stub_fallback):
    calc, committee = _make(uq_force=0.5, uq_energy=0.0)
    calc.calculate(_atoms(offset=0.0))
    calc.calculate(_atoms(offset=0.1))      # different geometry -> stub recomputes

    assert len(stub_fallback.instances) == 1
    assert stub_fallback.instances[0].calls == 2
    assert calc.n_calls == 2
    assert calc.n_fallback == 2
    assert committee.calls == 2


def test_force_only_rule_when_theta_energy_is_none(stub_fallback):
    calc, _ = _make(uq_force=0.01, uq_energy=1.0e6, theta_energy=None)
    calc.calculate(_atoms())
    assert calc.last_path == "nnp"          # a huge energy spread alone triggers nothing
    assert stub_fallback.instances == []


@pytest.mark.parametrize(
    "use_or_rule, uq_force, uq_energy, expected_fallback",
    [
        (True, 0.01, 0.01, False),
        (True, 0.50, 0.01, True),
        (True, 0.01, 0.50, True),
        (True, 0.50, 0.50, True),
        (False, 0.01, 0.01, False),
        (False, 0.50, 0.01, False),
        (False, 0.01, 0.50, False),
        (False, 0.50, 0.50, True),
    ],
)
def test_combined_rule_with_theta_energy(stub_fallback, use_or_rule, uq_force, uq_energy,
                                         expected_fallback):
    calc, _ = _make(
        uq_force=uq_force, uq_energy=uq_energy,
        theta_force=0.05, theta_energy=0.1, use_or_rule=use_or_rule,
    )
    calc.calculate(_atoms())
    assert (calc.last_path == "fallback") is expected_fallback
    assert calc.n_fallback == int(expected_fallback)
    assert len(stub_fallback.instances) == int(expected_fallback)


# --------------------------------------------------------------------------
# counters / statistics
# --------------------------------------------------------------------------
def test_counters_and_fallback_rate(stub_fallback):
    calc, _ = _make(uq_force=0.01, uq_energy=0.0)
    calc.calculate(_atoms(offset=0.0))
    calc.calculate(_atoms(offset=0.1))
    assert calc.stats() == {"n_calls": 2, "n_fallback": 0, "fallback_rate": 0.0}

    calc.committee.uq_force = 0.5           # next call falls back
    calc.calculate(_atoms(offset=0.2))
    assert calc.stats() == {
        "n_calls": 3,
        "n_fallback": 1,
        "fallback_rate": pytest.approx(1.0 / 3.0, rel=1e-12),
    }
    assert len(stub_fallback.instances) == 1


def test_ase_protocol_caches_both_properties_in_one_call(stub_fallback):
    calc, committee = _make(uq_force=0.01, uq_energy=0.0)
    atoms = _atoms()
    atoms.calc = calc
    atoms.get_potential_energy()
    atoms.get_forces()
    atoms.get_potential_energy()
    # energy and forces are produced by a single committee evaluation
    assert committee.calls == 1
    assert calc.n_calls == 1
