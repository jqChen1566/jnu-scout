"""SafeNNPCalculator: committee UQ with ORCA L2 fallback (ASE calculator protocol).

Mechanism
---------
1. Every committee member (MACE-MP-0 / MACE-MPA-0 / AIMNet2-b973c) evaluates the
   structure once; the spread gives the uncertainty uq.
2. uq < theta  -> use the committee ensemble mean (NNP path, fast).
3. uq >= theta -> fall back to ORCA L2 (r2SCAN-D4/def2-SVP, slow but reliable).

Relation to Staub et al. (2023, Molecules 28:4477): their xTB term is a smoothly
added physical guardrail (it participates in every prediction and prevents
geometry blow-up), with no run-time threshold or fallback; the fallback here is
our own mechanism, and the correctness of the calibrated theta is provided by
jnuscout.uq.calibration.

Units: ASE conventions (energy in eV, forces in eV/Å).
"""
from typing import Optional

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from jnuscout.calculators.committee import NNPCommittee
from jnuscout.calculators.orca import ORCACalculator


class SafeNNPCalculator(Calculator):
    """NNP calculator with a UQ-triggered fallback.

    Parameters
    ----------
    theta_force : float
        Force-disagreement threshold theta_F (eV/Å); obtained from the
        calibration theta = Quantile(uq_F | err > eps_crit, 5%).
    theta_energy : float or None
        Energy-disagreement threshold (eV/atom); None means the force criterion only.
    fallback_level : str
        Fallback method level, default "L2".
    committee : NNPCommittee or None
        Can be injected (testing / reuse).
    device : str
        "cuda" / "cpu".
    use_or_rule : bool
        If True (default), fall back when either criterion exceeds its threshold
        (OR rule, more conservative); if False, only when both do (AND rule).
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, theta_force: float, theta_energy: Optional[float] = None,
                 fallback_level: str = "L2", committee: Optional[NNPCommittee] = None,
                 device: str = "cuda", use_or_rule: bool = True,
                 fallback_kwargs: Optional[dict] = None):
        super().__init__()
        self.theta_force = float(theta_force)
        self.theta_energy = None if theta_energy is None else float(theta_energy)
        self.fallback_level = fallback_level
        self.committee = committee or NNPCommittee(device=device)
        self.use_or_rule = use_or_rule
        self._fallback_kwargs = fallback_kwargs or {}
        self._fallback = None  # built lazily (ORCA only starts up on a real fallback)

        # Counters (for coverage / fallback-rate auditing)
        self.n_calls = 0
        self.n_fallback = 0
        self.last_uq = None

    # ---------- decision rule ----------
    def _need_fallback(self, rep: dict) -> bool:
        over_f = rep["uq_force"] >= self.theta_force
        if self.theta_energy is None:
            return bool(over_f)
        over_e = rep["uq_energy"] >= self.theta_energy
        return bool(over_f or over_e) if self.use_or_rule else bool(over_f and over_e)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.n_calls += 1

        rep = self.committee.compute(atoms)
        self.last_uq = rep
        if self._need_fallback(rep):
            if self._fallback is None:
                self._fallback = ORCACalculator(level=self.fallback_level, **self._fallback_kwargs)
            at = atoms.copy()
            at.calc = self._fallback
            self.results = {
                "energy": float(at.get_potential_energy()),
                "forces": np.asarray(at.get_forces(), dtype=float),
            }
            self.n_fallback += 1
            self.last_path = "fallback"
        else:
            self.results = {
                "energy": rep["energy_mean"],
                "forces": rep["forces_mean"],
            }
            self.last_path = "nnp"

    # ---------- statistics ----------
    def stats(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "n_fallback": self.n_fallback,
            "fallback_rate": (self.n_fallback / self.n_calls) if self.n_calls else 0.0,
        }
