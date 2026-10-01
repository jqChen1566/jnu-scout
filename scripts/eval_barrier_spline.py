"""Barrier evaluation by the spline-endpoint method -- single- and multi-component.

Motivation: the single-component elementary-step pool of the Diels-Alder
library was nearly exhausted (95 available, about 92 used), so the benchmark
had to be extended with bimolecular entries (251 of them carry a spline).
An earlier evaluation script built the reactant from ``es["lhs"][0]``, which
is incorrect for bimolecular reactions (``path``/``idx_maps`` are empty and
the fragments cannot be concatenated).

This script uses the spline endpoints instead:

    reactant = spline.evaluate(0.0)          <- full reactant complex (correct relative orientation)
    TS       = spline.evaluate(ts_position)
    barrier  = (E_TS - E_reactant) kcal/mol

ORCA and every model see exactly the same two geometries, so the comparison
is fair across models.

Advantages: works for single- and multi-component entries alike; geometries
come from the Chemoton-fitted path, so the protocol is uniform.

Usage:
    python scripts/eval_barrier_spline.py --es-file data/holdout_bi.json \
        --model v3f=models/mace_al_v3f.model --out results/barrier_bi.json
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pymongo
import scine_database as db

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

BOHR = 1.8897259886
KCAL_PER_EV = 23.060548


def _orca_pair(task):
    """Worker: ORCA L2 single points for the (reactant, ts) pair."""
    es_id, els_r, pos_r, els_t, pos_t = task
    from ase import Atoms
    from jnuscout.calculators.orca import ORCACalculator
    orca = ORCACalculator(level="L2", nprocs=1, maxcore=2000)
    out = {"es_id": es_id}
    try:
        for tag, els, pos in [("reactant", els_r, pos_r), ("ts", els_t, pos_t)]:
            at = Atoms(els, positions=pos)
            at.calc = orca
            out[tag] = float(at.get_potential_energy())      # eV
        out["ok"] = True
    except Exception as e:
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def collect_pairs(dbname, es_list):
    """Collect the spline reactant / TS endpoint structures (Angstrom) for each ES."""
    mc = pymongo.MongoClient("127.0.0.1", 27017)[dbname]
    cred = db.Credentials()
    cred.hostname, cred.port, cred.database_name = "127.0.0.1", 27017, dbname
    mgr = db.Manager()
    mgr.set_credentials(cred)
    mgr.connect()
    mgr.init()
    col = mgr.get_collection("elementary_steps")
    prefix = {str(d["_id"])[:12]: d["_id"] for d in mc.elementary_steps.find({}, {"_id": 1})}

    pairs = []
    for item in es_list:
        eid = item["es_id"]
        oid = prefix.get(eid)
        if oid is None:
            continue
        es = db.ElementaryStep(db.ID(str(oid)), col)
        es.link(col)
        try:
            sp = es.get_spline()
            e_r, at_r = sp.evaluate(0.0)
            e_t, at_t = sp.evaluate(float(sp.ts_position))
        except Exception as e:
            print(f"  [skip] {eid}: {type(e).__name__}: {e}")
            continue
        els_r = [str(x) for x in at_r.elements]
        els_t = [str(x) for x in at_t.elements]
        if len(els_r) != len(els_t):
            print(f"  [skip] {eid}: inconsistent atom counts at the endpoints")
            continue
        # The component count must be read from the database: the ES list JSON
        # does not necessarily carry this field (an earlier version used
        # item.get("n_lhs", 1) and labeled every bimolecular entry as 1).
        es_doc = mc.elementary_steps.find_one({"_id": oid}, {"lhs": 1})
        n_lhs = len(es_doc.get("lhs", [])) if es_doc else 1
        pairs.append({
            "es_id": eid, "n_atoms": len(els_r),
            "n_lhs": n_lhs,
            "els": "".join(els_r),
            "pos_r": np.asarray(at_r.positions, dtype=float) / BOHR,
            "pos_t": np.asarray(at_t.positions, dtype=float) / BOHR,
            "spline_e_r": float(e_r), "spline_e_t": float(e_t),
        })
    return pairs


def make_calc(path, device="cuda"):
    if path.startswith("aimnet2:"):
        from aimnet2calc import AIMNet2ASE
        return AIMNet2ASE(base_calc=path.split(":", 1)[1], charge=0, mult=1)
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=[path], device=device, default_dtype="float64")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="")
    ap.add_argument("--es-file", required=True)
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--orca-cache", default="results/barrier_spline_orca_cache.json")
    ap.add_argument("--out", default="results/barrier_spline.json")
    args = ap.parse_args()

    spec = json.load(open(args.es_file))
    dbname = args.db or spec.get("db", "diels_alder")
    pairs = collect_pairs(dbname, spec["es"])
    print(f"available elementary steps: {len(pairs)} (target {len(spec['es'])})", flush=True)
    if not pairs:
        return 1
    from collections import Counter
    print(f"  atom-count distribution: {dict(sorted(Counter(p['n_atoms'] for p in pairs).items()))}")
    print(f"  component-count distribution: {dict(sorted(Counter(p['n_lhs'] for p in pairs).items()))}")

    # ---- ORCA (with cache) ----
    cache = json.load(open(args.orca_cache)) if os.path.exists(args.orca_cache) else {}
    todo = [p for p in pairs if p["es_id"] not in cache]
    print(f"ORCA evaluations needed: {len(todo)} (cache hits: {len(pairs)-len(todo)})", flush=True)
    if todo:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_orca_pair, (p["es_id"], p["els"], p["pos_r"],
                                           p["els"], p["pos_t"])): p for p in todo}
            for i, fut in enumerate(as_completed(futs)):
                r = fut.result()
                if r.get("ok"):
                    cache[r["es_id"]] = {"reactant": r["reactant"], "ts": r["ts"]}
                else:
                    print(f"  ORCA failed {r.get('es_id')}: {r.get('error')}")
                if (i + 1) % 5 == 0 or (i + 1) == len(todo):
                    print(f"  ORCA {i+1}/{len(todo)}", flush=True)
        os.makedirs(os.path.dirname(args.orca_cache) or ".", exist_ok=True)
        json.dump(cache, open(args.orca_cache, "w"), indent=1)
        print(f"ORCA wall time {time.time()-t0:.0f}s")

    # ---- models ----
    from ase import Atoms
    models = {}
    for s in args.model:
        name, path = s.split("=", 1)
        models[name] = make_calc(path, args.device)
        print(f"  loaded {name}")

    preds = {k: {} for k in models}
    for p in pairs:
        for k, cal in models.items():
            try:
                v = {}
                for tag, pos in [("reactant", p["pos_r"]), ("ts", p["pos_t"])]:
                    at = Atoms(p["els"], positions=pos)
                    at.calc = cal
                    v[tag] = float(at.get_potential_energy())
                preds[k][p["es_id"]] = v
            except Exception as e:
                print(f"  [{k}] {p['es_id']} failed: {type(e).__name__}: {e}")
    print()

    # ---- summary ----
    rows = []
    for p in pairs:
        eid = p["es_id"]
        if eid not in cache:
            continue
        row = {"es_id": eid, "n_atoms": p["n_atoms"], "n_lhs": p["n_lhs"],
               "orca": (cache[eid]["ts"] - cache[eid]["reactant"]) * KCAL_PER_EV}
        for k in models:
            if eid in preds[k]:
                row[k] = (preds[k][eid]["ts"] - preds[k][eid]["reactant"]) * KCAL_PER_EV
        rows.append(row)

    def stat(pr, rf):
        pr, rf = np.asarray(pr), np.asarray(rf)
        d = pr - rf
        if len(d) < 2:
            return {"n": len(d), "mae": float(np.abs(d).mean()), "bias": float(d.mean()),
                    "rho": None, "slope": None}
        a, _ = np.polyfit(rf, pr, 1)
        return {"n": len(d), "mae": float(np.abs(d).mean()), "bias": float(d.mean()),
                "rmse": float(np.sqrt((d ** 2).mean())),
                "rho": float(np.corrcoef(pr, rf)[0, 1]) if np.std(pr) > 0 else None,
                "slope": float(a)}

    ref = [r["orca"] for r in rows]
    names = [k for k in models if all(k in r for r in rows)] or list(models)
    result = {"db": dbname, "n_es": len(rows),
              "method": "spline endpoints (reactant = evaluate(0), TS = evaluate(ts_position))",
              "overall": {}, "by_n_lhs": {}}
    print("=" * 76)
    print(f"{'model':<14}{'group':<10}{'MAE':>9}{'bias':>9}{'rho':>8}{'slope':>8}{'n':>5}")
    print("=" * 76)
    for k in names:
        s = stat([r[k] for r in rows], ref)
        result["overall"][k] = s
        print(f"{k:<14}{'all':<10}{s['mae']:>9.2f}{s['bias']:>9.2f}"
              f"{(s['rho'] if s['rho'] is not None else float('nan')):>8.3f}"
              f"{(s['slope'] if s['slope'] is not None else float('nan')):>8.3f}{s['n']:>5}")
        for nl in sorted({r["n_lhs"] for r in rows}):
            sub = [r for r in rows if r["n_lhs"] == nl and k in r]
            if len(sub) < 2:
                continue
            ss = stat([r[k] for r in sub], [r["orca"] for r in sub])
            result["by_n_lhs"].setdefault(str(nl), {})[k] = ss
            print(f"{'':<14}{'%d-comp' % nl:<10}{ss['mae']:>9.2f}{ss['bias']:>9.2f}"
                  f"{(ss['rho'] if ss['rho'] is not None else float('nan')):>8.3f}"
                  f"{(ss['slope'] if ss['slope'] is not None else float('nan')):>8.3f}{ss['n']:>5}")
        print("-" * 76)

    print(f"\nORCA reference barrier range: {min(ref):.1f} - {max(ref):.1f} kcal/mol (mean {np.mean(ref):.1f})")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"summary": result, "rows": rows}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
