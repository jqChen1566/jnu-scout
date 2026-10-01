"""Validate the MACE fine-tuning: barrier accuracy on held-out elementary steps.

Compares three models on the same held-out ES with ORCA L2 as reference:
    before fine-tuning: MACE-MP-0 (foundation model)
    after fine-tuning:  mace_al_da (fine-tuned on Chemoton path configurations)
    reference:          ORCA L2

The held-out ES are extracted from the es_id fields of `data/al_valid2.xyz`,
which guarantees no overlap with the training set (split by ES, no path-point
leakage).

Usage:
    python scripts/validate_finetune.py --valid data/al_valid2.xyz \
        --finetuned models/mace_al_da.model --db diels_alder --out results/finetune_eval.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pymongo
from bson.objectid import ObjectId

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from jnuscout.calculators.orca import ORCACalculator

KCAL_PER_EV = 23.060548
BOHR_PER_ANGSTROM = 1.8897259886


def valid_es_ids(path):
    ids = set()
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            n = int(line)
            hdr = f.readline()
            for _ in range(n):
                f.readline()
            for tok in hdr.split():
                if tok.startswith("es_id="):
                    ids.add(tok.split("=", 1)[1])
    return ids


def get_struct(db, sid):
    """Return the elements and coordinates of a structure (always **in Angstrom**).

    Unit correction: SCINE `structures.atoms` stores coordinates in **Bohr**.
    Earlier versions fed them to ASE / ORCA (which expect Angstrom) without
    conversion, stretching every geometry by 1.8897x.  For one elementary step
    the ORCA barrier was 18.76 kcal/mol with the unconverted geometry and
    **90.76** kcal/mol once the conversion was applied.
    """
    s = db.structures.find_one({"_id": ObjectId(sid)})
    if not s:
        return None
    return ("".join(a["element"] for a in s["atoms"]),
            np.array([[a["x"], a["y"], a["z"]] for a in s["atoms"]],
                     dtype=float) / BOHR_PER_ANGSTROM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", default="data/al_valid2.xyz")
    ap.add_argument("--finetuned", default="models/mace_al_da.model")
    ap.add_argument("--baseline", default=None,
                    help="path to the MACE foundation model checkpoint (required), e.g. "
                         "the MACE-MP-0 medium model from the public mace-torch release")
    ap.add_argument("--db", default="diels_alder")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/finetune_eval.json")
    args = ap.parse_args()

    if args.baseline is None:
        ap.error("--baseline is required: path to the MACE foundation model checkpoint")

    ids = valid_es_ids(args.valid)
    print(f"Held-out ES (from {args.valid}): {len(ids)}")

    db = pymongo.MongoClient("127.0.0.1", 27017)[args.db]
    from ase import Atoms
    from mace.calculators import MACECalculator

    base_calc = MACECalculator(model_paths=[args.baseline], device=args.device, default_dtype="float64")
    ft_calc = MACECalculator(model_paths=[args.finetuned], device=args.device, default_dtype="float64")
    orca = ORCACalculator(level="L2")

    # es_id stores the 12-character hex prefix of an ObjectId.  $regex does not
    # work on ObjectId fields, so build a prefix -> ObjectId map first (the
    # database is small; a full scan is acceptable).
    prefix_map = {str(d["_id"])[:12]: d["_id"] for d in db.elementary_steps.find({}, {"_id": 1})}

    rows = []
    skipped_not_single = 0
    for eid in sorted(ids):
        oid = prefix_map.get(eid)
        if oid is None:
            print(f"  [skip] {eid} not found in the database")
            continue
        es = db.elementary_steps.find_one({"_id": oid})
        if es is None or len(es.get("lhs", [])) != 1 or len(es.get("rhs", [])) != 1:
            skipped_not_single += 1
            continue
        sr = get_struct(db, es["lhs"][0])
        st = get_struct(db, es["transition_state"])
        if sr is None or st is None or len(sr[0]) != len(st[0]):
            continue

        out = {"es_id": eid, "n_atoms": len(sr[0])}
        e = {}
        for tag, (els, pos) in [("reactant", sr), ("ts", st)]:
            at = Atoms(els, positions=pos)
            at.calc = base_calc
            eb = float(at.get_potential_energy())
            fb = np.asarray(at.get_forces())
            at2 = Atoms(els, positions=pos)
            at2.calc = ft_calc
            ef = float(at2.get_potential_energy())
            ff = np.asarray(at2.get_forces())
            at3 = Atoms(els, positions=pos)
            at3.calc = orca
            eo = float(at3.get_potential_energy())
            fo = np.asarray(at3.get_forces())
            e[tag] = {"base": eb, "ft": ef, "orca": eo}
            out[f"ferr_base_{tag}"] = float(np.linalg.norm(fb - fo, axis=1).mean())
            out[f"ferr_ft_{tag}"] = float(np.linalg.norm(ff - fo, axis=1).mean())

        out["barrier_base"] = (e["ts"]["base"] - e["reactant"]["base"]) * KCAL_PER_EV
        out["barrier_ft"] = (e["ts"]["ft"] - e["reactant"]["ft"]) * KCAL_PER_EV
        out["barrier_orca"] = (e["ts"]["orca"] - e["reactant"]["orca"]) * KCAL_PER_EV
        rows.append(out)
        print(f"  {eid} nAtoms={out['n_atoms']:2d} | ORCA {out['barrier_orca']:6.1f} | "
              f"baseline {out['barrier_base']:6.1f} | fine-tuned {out['barrier_ft']:6.1f} kcal/mol",
              flush=True)

    if not rows:
        print(f"No usable held-out ES ({skipped_not_single} not single-component rearrangements, "
              f"{len(ids)-skipped_not_single} not found)")
        return 1

    bo = np.array([r["barrier_orca"] for r in rows])
    bb = np.array([r["barrier_base"] for r in rows])
    bf = np.array([r["barrier_ft"] for r in rows])

    def st(pred):
        d = pred - bo
        return {"mae": float(np.abs(d).mean()), "bias": float(d.mean()),
                "rho": float(np.corrcoef(pred, bo)[0, 1]) if pred.std() > 0 else None}

    fb = np.mean([r["ferr_base_ts"] for r in rows])
    ff = np.mean([r["ferr_ft_ts"] for r in rows])
    summary = {
        "n_es": len(rows), "db": args.db,
        "baseline": st(bb), "finetuned": st(bf),
        "force_err_ts_base": float(fb), "force_err_ts_ft": float(ff),
        "improvement": {
            "mae_delta": float(abs(bb - bo).mean() - abs(bf - bo).mean()),
            "bias_delta": float(abs((bb - bo).mean()) - abs((bf - bo).mean())),
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    print("\n=== Fine-tuning result ===")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
