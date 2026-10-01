"""Active learning: build an ORCA-labelled training set from Chemoton reaction paths.

Rationale: NNP barrier heights carry a system-dependent systematic bias (a fitted
slope of 0.588 on the Diels-Alder benchmark, i.e. strong compression), and
ex-post linear corrections do not generalize across systems.  The principled
solution is to fine-tune on reaction-path configurations.

Procedure:
1. Pull elementary steps from the SCINE database and sample configurations along
   the MEP with `spline.evaluate(t)`.
2. Label energies + forces with ORCA L2 (r2SCAN-D4/def2-SVP), in parallel.
3. Write MACE training format (extxyz, eV / A).

Usage:
    python scripts/build_al_trainset.py --db diels_alder --n-points 15 --workers 8 \
        --out data/al_trainset_da.xyz
"""
import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pymongo
import scine_database as db

EV_PER_HARTREE = 27.211386245
BOHR_PER_ANGSTROM = 1.8897259886


def sample_ts_weights(n_points: int, ts_pos: float, ts_frac: float = 0.5,
                      window: float = 0.25) -> np.ndarray:
    """Weighted sampling around the TS: half of the sample points are
    concentrated within TS +/- window.

    Rationale: with uniform sampling too few points fall near the TS (with 15
    points only 1-2 land near the TS), while that region is exactly where the
    energy is most sensitive to the barrier and the forces are largest.  The
    force error at the TS therefore rose (0.61 -> 0.96 eV/A) after fine-tuning
    instead of falling.
    """
    n_ts = max(1, int(round(n_points * ts_frac)))
    lo, hi = max(0.0, ts_pos - window), min(1.0, ts_pos + window)
    ts_pts = np.linspace(lo, hi, n_ts)
    rest = np.linspace(0.0, 1.0, n_points - n_ts + 2)[1:-1]   # drop duplicate endpoints
    return np.unique(np.concatenate([ts_pts, rest, [0.0, 1.0]]))


def collect_frames(dbname: str, n_points: int, max_es: int,
                   holdout_es: int = 0, holdout: bool = False,
                   ts_weighted: bool = True) -> list:
    """Collect path configurations from the database.

    Returns (es_id, t, energy_Eh, elements, positions_bohr) per frame.

    When holdout_es > 0, the last holdout_es elementary steps form the holdout
    set (with ``holdout=True`` the holdout set itself is returned).

    The split must be done by elementary step, not by configuration: points along
    the same path are strongly correlated, and a random split by configuration
    would leak information and overestimate the fine-tuning benefit.
    """
    mc = pymongo.MongoClient("127.0.0.1", 27017)[dbname]
    cred = db.Credentials()
    cred.hostname, cred.port, cred.database_name = "127.0.0.1", 27017, dbname
    mgr = db.Manager()
    mgr.set_credentials(cred)
    mgr.connect()
    mgr.init()
    col = mgr.get_collection("elementary_steps")

    ids = [d["_id"] for d in mc.elementary_steps.find({"type": "regular"}) if d.get("spline")]
    if holdout_es > 0:
        if holdout:
            ids = ids[-holdout_es:]
        else:
            ids = ids[: len(ids) - holdout_es]
    ids = ids[:max_es]
    frames = []
    for eid in ids:
        es = db.ElementaryStep(db.ID(str(eid)), col)
        es.link(col)
        try:
            sp = es.get_spline()
        except Exception:
            continue
        els = [str(e) for e in sp.elements]
        ts_pos = float(sp.ts_position)
        ts = sample_ts_weights(n_points, ts_pos) if ts_weighted else np.linspace(0.0, 1.0, n_points)
        for t in ts:
            try:
                e_h, atoms = sp.evaluate(float(t))
                frames.append({
                    "es_id": str(eid)[:12], "t": float(t), "e_h": float(e_h),
                    "elements": els,
                    "positions_bohr": np.asarray(atoms.positions, dtype=float),
                })
            except Exception:
                continue
    return frames


def _orca_label(frame):
    """Label a single configuration with ORCA (subprocess): energy in eV + forces in eV/A."""
    from ase import Atoms
    from jnuscout.calculators.orca import ORCACalculator
    try:
        pos_ang = frame["positions_bohr"] / BOHR_PER_ANGSTROM
        at = Atoms("".join(frame["elements"]), positions=pos_ang)
        at.calc = ORCACalculator(level="L2", nprocs=1, maxcore=2000)
        e = float(at.get_potential_energy())      # eV
        f = np.asarray(at.get_forces())           # eV/A
        return {**{k: frame[k] for k in ("es_id", "t", "elements")},
                "energy_eV": e, "forces": f, "ok": True}
    except Exception as e:
        return {"es_id": frame.get("es_id"), "t": frame.get("t"),
                "ok": False, "error": f"{type(e).__name__}: {e}"}


def write_extxyz(path: str, labeled: list, ref_energy_eV: float = None):
    """Write an extxyz file readable by MACE (energies in eV)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n_written = 0
    with open(path, "w") as fh:
        for r in labeled:
            if not r.get("ok"):
                continue
            els, f, e = r["elements"], r["forces"], r["energy_eV"]
            pos = r["positions_bohr"] if "positions_bohr" in r else None
            # Forces and positions are in Angstrom
            fh.write(f"{len(els)}\n")
            fh.write(f'Properties=species:S:1:pos:R:3:forces:R:3 energy={e:.8f} '
                     f'es_id={r["es_id"]} t={r["t"]:.3f} pbc="F F F"\n')
            for i, el in enumerate(els):
                p = r["positions_ang"][i]
                fh.write(f"{el} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f} "
                         f"{f[i][0]:.8f} {f[i][1]:.8f} {f[i][2]:.8f}\n")
            n_written += 1
    return n_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="diels_alder",
                    help="comma-separated database names, e.g. diels_alder,h2co")
    ap.add_argument("--n-points", type=int, default=15,
                    help="number of sample points along the MEP per elementary step")
    ap.add_argument("--max-es", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="data/al_trainset.xyz")
    ap.add_argument("--holdout-es", type=int, default=0,
                    help="hold out N elementary steps for testing (split by ES to "
                         "avoid path-point leakage)")
    ap.add_argument("--holdout", action="store_true",
                    help="generate the holdout set instead of the training set")
    ap.add_argument("--uniform", action="store_true",
                    help="uniform sampling (default: TS-neighbourhood weighted)")
    args = ap.parse_args()

    t0 = time.time()
    frames = []
    for dbname in args.db.split(","):
        fr = collect_frames(dbname.strip(), args.n_points, args.max_es,
                            holdout_es=args.holdout_es, holdout=args.holdout,
                            ts_weighted=not args.uniform)
        print(f"  {dbname.strip()}: {len(fr)} frames", flush=True)
        frames.extend(fr)
    tag = "holdout set" if args.holdout else "training set"
    print(f"Collected {len(frames)} path configurations ({args.db}, {tag})", flush=True)
    if not frames:
        return 1

    # Convert to Angstrom for file output
    for fr in frames:
        fr["positions_ang"] = fr["positions_bohr"] * (1.0 / BOHR_PER_ANGSTROM)

    labeled, failed = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_orca_label, fr): fr for fr in frames}
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            if r.get("ok"):
                fr = futs[fut]
                r["positions_ang"] = fr["positions_ang"]
                labeled.append(r)
            else:
                failed += 1
            if (i + 1) % 20 == 0 or (i + 1) == len(frames):
                print(f"  labelling {i+1}/{len(frames)} (ok {len(labeled)}, failed {failed})",
                      flush=True)

    n = write_extxyz(args.out, labeled)
    print(f"\nWrote {n} frames -> {args.out}")
    print(f"Elapsed {time.time()-t0:.0f}s ({args.workers} parallel workers)")
    print("\nFine-tuning example:")
    print(f"  mace_run_train --name=mace_al_da --train_file={args.out} \\")
    print(f"      --foundation_model=<mace-mp-0.model> --device=cuda \\")
    print(f"      --E0s='average' --forces_weight=100 --energy_weight=1 \\")
    print(f"      --max_num_epochs=200 --batch_size=8 --lr=1e-4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
