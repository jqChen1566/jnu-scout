"""CalculatorFactory: return the L0/L1/L2/L3 calculators of the method chain.

L0  = SafeNNPCalculator (NNP committee + UQ-triggered ORCA L2 fallback)
L1  = XTBCalculator (GFN2-xTB via the xtb CLI; ASE 3.29 ships no built-in xTB
      calculator, hence the CLI wrapper)
L2  = ORCACalculator (r2SCAN-D4/def2-SVP)
L3  = ORCACalculator (wB97M-V/def2-TZVP)
"""
from ase.calculators.calculator import Calculator

from jnuscout.calculators.orca import ORCACalculator
from jnuscout.calculators.xtb import XTBCalculator


class CalculatorFactory:
    """Method-chain factory: L0 = Safe-NNP, L1 = GFN2-xTB, L2 = r2SCAN-D4/def2-SVP, L3 = wB97M-V/def2-TZVP."""

    @staticmethod
    def create(level, **kwargs):
        if level == "L0":
            # Thresholds must come from calibration (see scripts/calibrate_uq.py);
            # no defaults are provided here
            from jnuscout.calculators.safe_nnp import SafeNNPCalculator
            return SafeNNPCalculator(**kwargs)
        if level == "L1":
            return XTBCalculator(method=kwargs.pop("method", "GFN2-xTB"), **kwargs)
        if level in ("L2", "L3"):
            return ORCACalculator(level=level, **kwargs)
        raise ValueError(f"unknown level: {level}")
