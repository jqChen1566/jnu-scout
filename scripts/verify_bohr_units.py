"""Check that geometries from the search database are converted from Bohr to
Angstrom correctly.

Background: the SCINE database stores ``structures.atoms`` x/y/z in Bohr
(atomic units).  Measured example: the nearest-neighbour distance of an n=6
reactant is 2.048, which divided by 1.8897 gives 1.084 Angstrom (a C-H bond).

Code paths that returned the raw coordinates without the Bohr -> Angstrom
conversion and fed them to ASE ``Atoms(positions=...)`` (which expects
Angstrom) or to ORCA would stretch every geometry by a factor of 1.8897
(C-H 1.08 Angstrom -> 2.05 Angstrom, already a broken-bond distance).

This script evaluates the same elementary step with both geometry
conventions and quantifies the effect on the ORCA barrier:

    A) raw coordinates used as if they were Angstrom (the convention to avoid)
    B) divided by 1.8897259886 (correct)

Usage:
    python scripts/verify_bohr_units.py --es <es-id> --out results/bohr_unit_check.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pymongo

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from jnuscout.calculators.orca import ORCACalculator

BOHR = 1.8897259886
KCAL = 23.060548


def get_struct(db, sid):
    from bson.objectid import ObjectId
    s = db.structures.find_one({"_id": ObjectId(sid)})
    if not s:
        return None
    return ("".join(a["element"] for a in s["atoms"]),
            np.array([[a["x"], a["y"], a["z"]] for a in s["atoms"]], dtype=float))


def min_bond(pos):
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, 9e9)
    return float(d.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="diels_alder")
    ap.add_argument("--es", required=True, help="elementary-step id (12-char ObjectId prefix)")
    ap.add_argument("--model", default="")
    ap.add_argument("--out", default="results/bohr_unit_check.json")
    args = ap.parse_args()

    mc = pymongo.MongoClient("127.0.0.1", 27017)[args.db]
    prefix = {str(d["_id"])[:12]: d["_id"] for d in mc.elementary_steps.find({}, {"_id": 1})}
    oid = prefix[args.es]
    es = mc.elementary_steps.find_one({"_id": oid})

    from ase import Atoms
    orca = ORCACalculator(level="L2")
    mace = None
    if args.model:
        from mace.calculators import MACECalculator
        mace = MACECalculator(model_paths=[args.model], device="cuda",
                              default_dtype="float64")
        print(f"comparison model: {args.model}")

    variants = {"A_bohr_as_angstrom": 1.0, "B_correct": 1.0 / BOHR}
    res = {"es_id": args.es, "n_atoms": None, "variants": {}}

    for vname, scale in variants.items():
        row = {}
        for tag, sid in [("reactant", es["lhs"][0]), ("ts", es["transition_state"])]:
            els, pos = get_struct(mc, sid)
            pos = pos * scale
            at = Atoms(els, positions=pos)
            entry = {"min_bond_A": min_bond(pos)}
            t0 = time.time()
            at.calc = orca
            entry["orca_E"] = float(at.get_potential_energy())
            if mace is not None:
                at2 = Atoms(els, positions=pos)
                at2.calc = mace
                entry["mace_E"] = float(at2.get_potential_energy())
            entry["t"] = round(time.time() - t0, 1)
            row[tag] = entry
            print(f"  [{vname}] {tag}: min_bond={entry['min_bond_A']:.3f} A "
                  f"ORCA={entry['orca_E']:.6f} Eh ({entry['t']}s)", flush=True)
        row["barrier_orca_kcal"] = (row["ts"]["orca_E"] - row["reactant"]["orca_E"]) * KCAL
        if mace is not None:
            row["barrier_mace_kcal"] = (row["ts"]["mace_E"] - row["reactant"]["mace_E"]) * KCAL
        res["variants"][vname] = row
        res["n_atoms"] = len(els)

    print("\n" + "=" * 66)
    a, b = res["variants"]["A_bohr_as_angstrom"], res["variants"]["B_correct"]
    print(f"ES {args.es} ({res['n_atoms']} atoms)")
    print(f"  shortest bond : A {a['reactant']['min_bond_A']:.3f} A  ->  B {b['reactant']['min_bond_A']:.3f} A")
    print(f"  ORCA barrier  : A {a['barrier_orca_kcal']:8.2f}  ->  B {b['barrier_orca_kcal']:8.2f} kcal/mol")
    if mace is not None:
        print(f"  MACE barrier  : A {a['barrier_mace_kcal']:8.2f}  ->  B {b['barrier_mace_kcal']:8.2f} kcal/mol")
        print(f"  error (A)     : {a['barrier_mace_kcal']-a['barrier_orca_kcal']:+.2f}  |  "
              f"error (B): {b['barrier_mace_kcal']-b['barrier_orca_kcal']:+.2f} kcal/mol")
    print("=" * 66)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
