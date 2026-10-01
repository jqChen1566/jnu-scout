"""Multi-foundation-model committee (MACE-MP-0 + MACE-MPA-0 + AIMNet2(b973c)) plus disagreement metrics.

Member selection rationale (see docs/architecture.md):
- The AIMNet2 wb97m_0 weights show a force-directionality defect: for 3 of 5 test
  molecules the forces are antiparallel or orthogonal to the ORCA reference
  (for H2O, cos = 0.0006).
- The b973c weights are directionally correct (cos >= 0.9995); the residual
  magnitude deviation of 28-44% is a systematic inter-functional difference that
  UQ calibration can absorb.
- The committee therefore uses AIMNet2(b973c); the example configurations in
  ``configs/`` follow the same choice.

UQ metrics (defined here; shared by SafeNNPCalculator and the calibration scripts):
- Energy disagreement  uq_E = std_i(E_i) / N             [eV/atom]
    Divided by the number of atoms to remove the size dependence.
- Force disagreement   uq_F = max_j max_i ||F_i,j - F_mean_j||   [eV/Å]
    F_mean_j is the committee-mean force on atom j; uq_F is the largest per-atom
    deviation (conservative: a local anomaly is not averaged away).

The force disagreement takes a max rather than a mean because failures along an
AFIR path often appear as a wrong force direction on a single atom; averaging
would dilute them with the remaining well-behaved atoms.
"""
import logging
import os
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _default_members():
    """Default committee members: name -> (backend, model identifier).

    Model identifiers are preferably taken from environment variables (on an
    offline machine they must point to local weight files, otherwise MACE will
    try to download them):
      MACE_MP0_PATH  -- MACE-MP-0 weights (.model path); default "medium"
                        (resolved online)
      MACE_MPA0_PATH -- MACE-MPA-0 weights; default
                        ~/.cache/mace/mace-mpa-0-medium.model
    """
    mace_mp0 = os.environ.get("MACE_MP0_PATH", "medium")
    mace_mpa0 = os.environ.get(
        "MACE_MPA0_PATH",
        os.path.expanduser("~/.cache/mace/mace-mpa-0-medium.model"),
    )
    return [
        ("MACE-MP-0", "mace", mace_mp0),
        ("MACE-MPA-0", "mace", mace_mpa0),
        ("AIMNet2-b973c", "aimnet2", "aimnet2_b973c"),
    ]


DEFAULT_MEMBERS = _default_members()


class NNPCommittee:
    """Multi-foundation-model committee: predictions plus disagreement metrics.

    Parameters
    ----------
    members : list of (name, backend, model_id)
        Defaults to DEFAULT_MEMBERS.
    device : str
        "cuda" or "cpu" (passed to the backend calculators).
    """

    def __init__(self, members=None, device: str = "cuda",
                 energy_baselines: Optional[Dict[str, float]] = None,
                 energy_family: Optional[str] = "mace"):
        self.members = list(members) if members is not None else list(DEFAULT_MEMBERS)
        self.device = device
        self._calcs: Dict[str, object] = {}
        self.last_report: Optional[dict] = None
        # Backend family whose members enter the energy disagreement (families use
        # different energy definitions, see disagreement()); None = pick the largest
        # family automatically.
        self.energy_family = energy_family
        # Energy baseline offsets (eV): the foundation models use different energy
        # references.  Measured for H2O: MACE-MP-0 -14.15, MACE-MPA-0 -13.78,
        # AIMNet2-b973c -2078.91 eV; computing the disagreement from raw absolute
        # energies would give a meaningless 2000+ eV value (uq_energy 324 eV/atom).
        # Estimated by calibrate_baselines() on a common set of configurations
        # (differences of the per-model mean energies), or supplied externally.
        self.energy_baselines: Dict[str, float] = dict(energy_baselines or {})

    # ---------- backend construction (lazy) ----------
    def _get_calc(self, name: str, backend: str, model_id: str):
        if name in self._calcs:
            return self._calcs[name]
        if backend == "mace":
            from mace.calculators import mace_mp
            # mace_mp's model argument accepts "medium" (MP-0) or a local .model path
            if model_id.endswith(".model"):
                calc = mace_mp(model=model_id, device=self.device, default_dtype="float64")
            else:
                calc = mace_mp(model=model_id, device=self.device, default_dtype="float64")
        elif backend == "aimnet2":
            from aimnet2calc import AIMNet2ASE
            calc = AIMNet2ASE(base_calc=model_id, charge=0, mult=1)
        else:
            raise ValueError(f"unknown backend: {backend}")
        self._calcs[name] = calc
        return calc

    # ---------- single prediction ----------
    def predict_members(self, atoms) -> Dict[str, dict]:
        """Predict with every member independently.

        Returns
        -------
        dict : name -> {"energy": float (eV), "forces": ndarray (N,3) (eV/Å)}
        """
        out = {}
        for name, backend, model_id in self.members:
            calc = self._get_calc(name, backend, model_id)
            at = atoms.copy()
            at.calc = calc
            out[name] = {
                "energy": float(at.get_potential_energy()),
                "forces": np.asarray(at.get_forces(), dtype=float),
            }
        return out

    # ---------- baseline calibration ----------
    def calibrate_baselines(self, atoms_list) -> Dict[str, float]:
        """Estimate each member's constant energy offset on a common set of configurations.

        For each member, take the mean energy over the batch; with the first member
        as reference, offset_i = mean_i - mean_0.  The offset is assumed constant
        (the dominant part of the difference between functionals / training sets
        appears as a constant term); only after this alignment do the inter-member
        energy differences reflect a genuine disagreement.

        Note: this is an approximate alignment -- if a model's systematic bias
        varies with the configuration, the residual leaks into uq_energy.  The
        force disagreement is unaffected (forces are energy derivatives, so constant
        offsets cancel); this is also why the force criterion is the default.
        """
        sums: Dict[str, list] = {n: [] for n, _, _ in self.members}
        for atoms in atoms_list:
            preds = self.predict_members(atoms)
            for n in preds:
                sums[n].append(preds[n]["energy"] / max(len(atoms), 1))  # per-atom energy
        means = {n: float(np.mean(v)) for n, v in sums.items() if v}
        if not means:
            raise ValueError("baseline calibration failed: no valid predictions")
        base = means[self.members[0][0]]
        # Store a per-atom offset (eV/atom): the measured offset scales with the
        # system size (total offset ~2065 eV for H2O, ~1079 eV for CH4), so a single
        # constant total-energy offset cancels only part of it; scaling by the atom
        # count is what aligns the references.
        self.energy_baselines = {n: means[n] - base for n in means}
        return self.energy_baselines

    # ---------- disagreement metrics ----------
    def disagreement(self, predictions: Dict[str, dict]) -> dict:
        """Compute the committee disagreement (uq_E, uq_F) and the ensemble mean.

        uq_E uses baseline-aligned energies (see energy_baselines); without
        calibration it degenerates to the raw absolute-energy disagreement, which
        has no physical meaning (the models use different energy references) and is
        meant for diagnostics only.

        Degenerate inputs are reported via ``logging`` warnings, not errors:
        prediction keys that do not match the committee members, an
        ``energy_family`` without available members, and members missing from
        ``energy_baselines`` all leave the computed values unchanged but emit a
        warning.
        """
        names = list(predictions)
        if len(names) < 2:
            raise ValueError("committee needs at least 2 members to compute a disagreement")
        n_atoms = predictions[names[0]]["forces"].shape[0]
        energies_raw = np.array([predictions[n]["energy"] for n in names])
        # The baseline offset is a per-atom quantity, so scale it by the atom count
        energies = np.array([
            predictions[n]["energy"] - self.energy_baselines.get(n, 0.0) * n_atoms for n in names
        ])
        forces = np.stack([predictions[n]["forces"] for n in names])  # (M, N, 3)

        # Energy disagreement: computed among members of the same backend family
        # only.  The foundation models use different energy definitions -- MACE
        # reports atomization energies (H2O -14.1 eV) while AIMNet2 reports absolute
        # total energies (H2O -2078.9 eV, close to -2078.4 eV from ORCA wB97M-V);
        # subtracting across families is not physically meaningful.  By default the
        # largest backend family is used (mace: MP-0 + MPA-0).
        # The family filter below is keyed by member name: prediction keys that
        # do not match ``self.members`` silently degrade it.  Warn, but keep the
        # deliberately tolerant fallback behaviour unchanged.
        member_names = [n for n, _, _ in self.members]
        missing = [n for n in member_names if n not in predictions]
        unknown = [n for n in names if n not in set(member_names)]
        if missing or unknown:
            logger.warning(
                "committee disagreement: prediction keys do not match the committee "
                "members (members without predictions: %s; predictions without a member "
                "entry: %s); the backend-family filter may silently degrade",
                missing, unknown,
            )

        fam_map: Dict[str, list] = {}
        for name, backend, _ in self.members:
            if name in predictions:
                fam_map.setdefault(backend, []).append(name)
        if self.energy_family and self.energy_family in fam_map:
            e_names = fam_map[self.energy_family]
        elif fam_map:
            if self.energy_family:
                logger.warning(
                    "committee disagreement: energy_family %r has no members among the "
                    "predictions (available families: %s); using the largest family "
                    "instead", self.energy_family, sorted(fam_map),
                )
            e_names = max(fam_map.values(), key=len)
        else:
            logger.warning(
                "committee disagreement: no member matched the predictions; uq_energy "
                "compares all %d predictions regardless of backend family, so their "
                "energy references may be incompatible", len(names),
            )
            e_names = names
        e_idx = [names.index(n) for n in e_names]
        missing_baselines = [n for n in e_names if n not in self.energy_baselines]
        if self.energy_baselines and missing_baselines:
            logger.warning(
                "committee disagreement: no energy baseline for %s (energy_baselines "
                "covers %s); their raw absolute-energy offsets leak into uq_energy",
                missing_baselines, sorted(self.energy_baselines),
            )
        uq_e = float(np.std(energies[e_idx]) / max(n_atoms, 1))
        # Force disagreement: for each atom take the largest deviation from the mean
        # over the members, then take the maximum over all atoms
        f_mean = forces.mean(axis=0)                       # (N, 3)
        dev = np.linalg.norm(forces - f_mean[None, :, :], axis=2)  # (M, N)
        uq_f = float(dev.max())

        # Ensemble energy: mean of the aligned energies, with the reference member's
        # original offset added back (keeps the absolute energy physical)
        ref_name = names[0]
        energy_ens = (float(energies.mean()) + self.energy_baselines.get(ref_name, 0.0) * n_atoms) \
            if self.energy_baselines else float(energies_raw.mean())

        return {
            "uq_energy": uq_e,
            "uq_force": uq_f,
            "energy_mean": energy_ens,                    # eV (for ASE)
            "forces_mean": f_mean,                        # (N,3) eV/Å
            "energy_spread": float(energies.max() - energies.min()),  # after alignment
            "energy_spread_raw": float(energies_raw.max() - energies_raw.min()),
            "baselines_aligned": bool(self.energy_baselines),
            "per_member_energy": {n: predictions[n]["energy"] for n in names},
        }

    def compute(self, atoms) -> dict:
        """Full pipeline: per-member predictions, then disagreement.  The result is also stored in last_report."""
        preds = self.predict_members(atoms)
        rep = self.disagreement(preds)
        rep["per_member_forces"] = {n: preds[n]["forces"] for n in preds}
        self.last_report = rep
        return rep
