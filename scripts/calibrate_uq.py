"""UQ threshold calibration experiment.

Workflow:
1. Build a reference set: N configurations (small CHNO organic molecules plus
   randomly displaced configurations).
2. For each configuration compute:
   - ORCA L2 (r2SCAN-D4/def2-SVP) energy + forces  ->  reference ground truth
   - committee (MACE-MP-0 / MPA-0 / AIMNet2-b973c) ->  predictions + disagreement UQ
   - true error err = per-atom force error (eV/A): mean_j ||F_nnp,j - F_ref,j||
3. Calibrate theta_F = Quantile(uq_F | err > eps_crit, 5%).
4. Report: Spearman rho, ECE, coverage / fallback rate / miss rate, NNP-vs-L2 MAE.

Acceptance criteria: rho > 0.6; coverage > 80%; fallback rate < 20%.

Usage:
    python scripts/calibrate_uq.py --n 60 --epsilon 0.10 --out results/uq_calib.json
"""
import argparse
import json
import os
import sys
import time
import traceback
from itertools import product

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from jnuscout.calculators.committee import NNPCommittee
from jnuscout.calculators.orca import ORCACalculator
from jnuscout.uq.calibration import (
    calibrate_threshold, spearman_rho, expected_calibration_error, coverage_stats,
)


# ---------- Reference set: small CHNO organic molecules ----------
MOLECULES = {
    "H2O":   ("OH2",  [[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]]),
    "CH4":   ("CH4",  [[0, 0, 0], [0.6291, 0.6291, 0.6291], [-0.6291, -0.6291, 0.6291],
                       [-0.6291, 0.6291, -0.6291], [0.6291, -0.6291, -0.6291]]),
    "NH3":   ("NH3",  [[0, 0, 0.117], [0, 0.939, -0.273], [0.813, -0.469, -0.273], [-0.813, -0.469, -0.273]]),
    "H2CO":  ("CH2O", [[0, 0, 0], [1.2, 0, 0], [-0.545, 0.944, 0], [-0.545, -0.944, 0]]),
    "C2H4":  ("C2H4", [[0.6695, 0, 0], [-0.6695, 0, 0], [1.2321, 0.9289, 0],
                       [1.2321, -0.9289, 0], [-1.2321, 0.9289, 0], [-1.2321, -0.9289, 0]]),
    "CH3OH": ("CH4O", [[0, 0, 0], [1.4, 0, 0], [-0.4, 0.9, 0], [-0.4, -0.9, 0],
                       [1.8, 0.9, 0], [1.8, -0.5, 0.8]]),
    "C2H2":  ("C2H2", [[0, 0, 0.6], [0, 0, -0.6], [0, 0, 1.66], [0, 0, -1.66]]),
    "HCN":   ("CHN",  [[0, 0, 0], [0, 0, 1.16], [0, 0, -1.06]]),
    "HCOOH":  ("CH2O2", [[0, 0, 0], [1.20, 0.10, 0], [-0.62, 0.85, 0], [1.75, 0.90, 0], [-1.05, -0.62, 0]]),
    "C2H6":  ("C2H6", [[0, 0, 0], [1.53, 0, 0], [-0.4, 1.0, 0], [-0.4, -0.5, 0.87],
                       [-0.4, -0.5, -0.87], [1.93, 1.0, 0], [0.93, 1.0, 0.87], [0.93, 1.0, -0.87]]),
}


def build_reference_set(n_per_mol: int, seed: int = 20260910, sigma: float = 0.05):
    """Build the reference set: n random displaced configurations per molecule.

    The displacement magnitude must keep every configuration chemically
    reasonable: a random displacement with sigma = 0.15 A produces many
    configurations with overlapping atoms (ORCA reference forces reach
    127-446 eV/A), and NNP failure on such configurations is unavoidable
    rather than detectable -- yet they never occur in a real AFIR search.
    Mixing them in inflates the calibrated threshold theta and distorts the
    coverage.

    The default sigma = 0.05 A corresponds to near-equilibrium thermal
    fluctuation amplitudes.

    Parameters
    ----------
    n_per_mol : int
    sigma : float
        Standard deviation of the per-component Gaussian displacement (A).
    """
    from ase import Atoms
    rng = np.random.default_rng(seed)
    cases = []
    for name, (formula, pos) in MOLECULES.items():
        base = Atoms(formula, positions=pos)
        cases.append((f"{name}_eq", base))
        for k in range(n_per_mol):
            at = base.copy()
            disp = rng.normal(scale=sigma, size=at.positions.shape)
            at.positions = at.positions + disp
            cases.append((f"{name}_d{k}", at))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5,
                    help="number of displaced configurations per molecule")
    ap.add_argument("--sigma", type=float, default=0.05,
                    help="displacement standard deviation (A); 0.05 = near-equilibrium "
                         "thermal noise, 0.15 produces unphysical overlapping configurations")
    ap.add_argument("--epsilon", type=float, default=0.10,
                    help="danger threshold eps_crit (eV/A, per-atom force error)")
    ap.add_argument("--theta-quantile", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/uq_calib.json")
    ap.add_argument("--limit", type=int, default=0,
                    help="only run the first N configurations when > 0 (debug)")
    ap.add_argument("--calibrate-baselines", action="store_true",
                    help="first align the per-member energy baselines on the same batch of "
                         "configurations (different foundation models use different energy references)")
    ap.add_argument("--no-orca", action="store_true",
                    help="skip the ORCA reference (only measure committee timing / "
                         "disagreement distribution, no threshold calibration)")
    args = ap.parse_args()

    cases = build_reference_set(args.n, sigma=args.sigma)
    if args.limit:
        cases = cases[: args.limit]
    print(f"Reference set: {len(cases)} configurations | eps_crit = {args.epsilon} eV/A | "
          f"device={args.device}", flush=True)

    committee = NNPCommittee(device=args.device)
    ref_calc = ORCACalculator(level="L2")

    # Energy baseline calibration (different foundation models use different energy
    # references: AIMNet2 and MACE differ by ~2065 eV in practice).
    if args.calibrate_baselines:
        bl_atoms = [a for _, a in cases[: max(4, len(cases) // 5)]]
        baselines = committee.calibrate_baselines(bl_atoms)
        print("Energy baseline offsets:", {k: round(v, 3) for k, v in baselines.items()},
              "eV", flush=True)

    rows, failures = [], []
    t0 = time.time()
    for i, (name, atoms) in enumerate(cases):
        try:
            # Reference ground truth (ORCA L2)
            if args.no_orca:
                f_ref, e_ref = None, None
            else:
                at_ref = atoms.copy(); at_ref.calc = ref_calc
                f_ref = np.asarray(at_ref.get_forces(), dtype=float)
                e_ref = float(at_ref.get_potential_energy())
            # Committee
            at_c = atoms.copy(); at_c.calc = None
            rep = committee.compute(at_c)
            f_nnp = rep["forces_mean"]
            err = float(np.linalg.norm(f_nnp - f_ref, axis=1).mean()) if f_ref is not None else float("nan")
            rows.append({
                "name": name, "n_atoms": len(atoms),
                "uq_force": rep["uq_force"], "uq_energy": rep["uq_energy"],
                "err_force": err,
                # Note: e_ref (ORCA absolute total energy) and e_nnp (MACE-family
                # atomization energy) use different baselines; their difference is
                # meaningless and kept for diagnostics only -- no error statistics
                # are derived from it.
                # TODO: comparing energies requires converting the MACE atomization
                # energy into an absolute energy (adding isolated-atom references)
                # or switching to relative energies.
                "e_ref": e_ref, "e_nnp": rep["energy_mean"],
                "energy_spread": rep["energy_spread"],
                "energy_spread_raw": rep["energy_spread_raw"],
            })
            print(f"[{i+1}/{len(cases)}] {name:12s} uq_F={rep['uq_force']:.4f} err={err:.4f} "
                  f"| uq_E={rep['uq_energy']:.4f} eV/atom", flush=True)
        except Exception as e:  # a single failed configuration must not abort the batch
            failures.append({"name": name, "error": f"{type(e).__name__}: {e}",
                             "tb": traceback.format_exc()[-500:]})
            print(f"[{i+1}/{len(cases)}] {name:12s} FAILED: {e}", flush=True)

    if not rows:
        print("No valid results; exiting", file=sys.stderr)
        return 1

    uq = np.array([r["uq_force"] for r in rows])
    err = np.array([r["err_force"] for r in rows])

    theta = calibrate_threshold(uq, err, args.epsilon, args.theta_quantile)
    rho = spearman_rho(uq, err)
    ece = expected_calibration_error(uq, err, args.epsilon)
    cov = coverage_stats(uq, err, theta, args.epsilon)

    summary = {
        "n_cases": len(rows), "n_failures": len(failures),
        "epsilon_crit": args.epsilon,
        "sigma": args.sigma,
        "theta_force": theta,
        "spearman_rho": rho,
        "ece": ece,
        "coverage": cov,
        "mae_force_nnp": float(err.mean()),
        "wall_time_s": time.time() - t0,
        "verdict": {
            "rho_gt_0.6": bool(rho > 0.6),
            "coverage_gt_80pct": bool(cov["coverage"] > 0.80),
            "fallback_lt_20pct": bool(cov["fallback_rate"] < 0.20),
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": rows, "failures": failures}, f, indent=1)

    print("\n=== Calibration result ===")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
