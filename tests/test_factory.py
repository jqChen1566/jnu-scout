"""Characterization tests for ``CalculatorFactory`` level routing (ASE only).

Routing table under test:
    L0 -> SafeNNPCalculator (NNP committee + UQ-triggered ORCA fallback)
    L1 -> XTBCalculator     (GFN2-xTB via the CLI wrapper)
    L2 -> ORCACalculator    (r2SCAN-D4/def2-SVP)
    L3 -> ORCACalculator    (wB97M-V/def2-TZVP)

Constructing the calculators never runs a backend: xTB/ORCA are only invoked
by ``calculate()``.
"""
import pytest

pytest.importorskip("ase")

from jnuscout.calculators.committee import NNPCommittee  # noqa: E402
from jnuscout.calculators.factory import CalculatorFactory  # noqa: E402
from jnuscout.calculators.orca import ORCA_KEYWORDS, ORCACalculator  # noqa: E402
from jnuscout.calculators.safe_nnp import SafeNNPCalculator  # noqa: E402
from jnuscout.calculators.xtb import XTBCalculator  # noqa: E402


# --------------------------------------------------------------------------
# L0
# --------------------------------------------------------------------------
def test_l0_returns_safe_nnp_calculator():
    calc = CalculatorFactory.create("L0", theta_force=0.07)
    assert isinstance(calc, SafeNNPCalculator)
    assert calc.theta_force == 0.07
    assert calc.theta_energy is None
    assert calc.fallback_level == "L2"
    assert calc.use_or_rule is True
    assert isinstance(calc.committee, NNPCommittee)
    assert calc._fallback is None  # ORCA is built lazily, on the first fallback
    assert calc.stats() == {"n_calls": 0, "n_fallback": 0, "fallback_rate": 0.0}


def test_l0_forwards_optional_thresholds_and_rule():
    calc = CalculatorFactory.create(
        "L0", theta_force=0.1, theta_energy=0.02, fallback_level="L3",
        use_or_rule=False, fallback_kwargs={"nprocs": 4},
    )
    assert calc.theta_energy == 0.02
    assert calc.fallback_level == "L3"
    assert calc.use_or_rule is False
    assert calc._fallback_kwargs == {"nprocs": 4}


def test_l0_requires_theta_force():
    # thresholds must come from calibration -- the factory provides no default
    with pytest.raises(TypeError):
        CalculatorFactory.create("L0")


# --------------------------------------------------------------------------
# L1
# --------------------------------------------------------------------------
def test_l1_returns_xtb_calculator_with_default_method():
    calc = CalculatorFactory.create("L1")
    assert isinstance(calc, XTBCalculator)
    assert calc.method == "GFN2-xTB"


def test_l1_method_kwarg_is_consumed_by_the_factory():
    calc = CalculatorFactory.create("L1", method="GFN1-xTB")
    assert calc.method == "GFN1-xTB"


def test_l1_forwards_charge_and_mult():
    calc = CalculatorFactory.create("L1", charge=-1, mult=2)
    assert calc.charge == -1
    assert calc.mult == 2


def test_l1_rejects_unknown_kwargs():
    with pytest.raises(TypeError):
        CalculatorFactory.create("L1", bogus_kwarg=1)


# --------------------------------------------------------------------------
# L2 / L3
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "level, marker, basis",
    [("L2", "r2SCAN", "def2-SVP"), ("L3", "wB97M-V", "def2-TZVP")],
)
def test_l2_and_l3_return_orca_calculators(level, marker, basis):
    calc = CalculatorFactory.create(level)
    assert isinstance(calc, ORCACalculator)
    assert calc.level == level
    assert calc.keywords == ORCA_KEYWORDS[level]
    assert marker in calc.keywords
    assert basis in calc.keywords


def test_l2_forwards_resource_kwargs():
    calc = CalculatorFactory.create("L2", nprocs=4, maxcore=2000)
    assert calc.nprocs == 4
    assert calc.maxcore == 2000


# --------------------------------------------------------------------------
# error behaviour / call convention
# --------------------------------------------------------------------------
@pytest.mark.parametrize("level", ["L4", "l0", "l1", "L2 ", "", None, 0, 2])
def test_unknown_level_raises_value_error(level):
    with pytest.raises(ValueError, match="unknown level"):
        CalculatorFactory.create(level)


def test_unknown_level_raises_even_with_kwargs():
    with pytest.raises(ValueError, match="unknown level"):
        CalculatorFactory.create("L9", theta_force=0.1)


def test_create_is_a_static_method():
    assert isinstance(CalculatorFactory.__dict__["create"], staticmethod)
    # callable both on the class and on an instance
    assert isinstance(CalculatorFactory.create("L1"), XTBCalculator)
    assert isinstance(CalculatorFactory().create("L1"), XTBCalculator)
