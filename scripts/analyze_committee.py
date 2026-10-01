"""Committee combination analysis (correct-unit version).

On corrected geometries the ranking of the committee members reverses relative
to an earlier evaluation on unconverted geometries:
    unconverted geometries: MP0 6.33 < AIMNet2 7.12 < MPA0 9.59
    corrected geometries:   AIMNet2 7.20 < MPA0 11.83 < MP0 21.69

Moreover the AIMNet2 bias has the opposite sign to the MACE-family members
(+5.90 vs -10.70/-20.40), which directly affects committee design: biases may
cancel, or one member may dominate the combined error.

This script compares member combinations to answer whether the equal-weight
committee is still the right choice.

Usage:
    python scripts/analyze_committee.py --in results/barrier_committee.json
"""
import argparse
import json
import itertools
import sys

import numpy as np


def st(pred, ref):
    pred, ref = np.asarray(pred), np.asarray(ref)
    d = pred - ref
    a, b = np.polyfit(ref, pred, 1)
    return {"n": len(d), "mae": float(np.abs(d).mean()), "bias": float(d.mean()),
            "rmse": float(np.sqrt((d ** 2).mean())),
            "rho": float(np.corrcoef(pred, ref)[0, 1]),
            "slope": float(a), "r2": float(np.corrcoef(pred, ref)[0, 1] ** 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/barrier_committee.json")
    ap.add_argument("--neg-thresh", type=float, default=-1.0)
    args = ap.parse_args()

    d = json.load(open(args.inp))
    rows = [r for r in d["rows"] if r["orca"] > args.neg_thresh]
    members = [k for k in d["rows"][0] if k not in ("es_id", "n_atoms", "orca")]
    ref = [r["orca"] for r in rows]
    print(f"Valid ES: {len(rows)} (of {len(d['rows'])}) | members: {members}\n")

    print("=" * 74)
    print("[Single members]")
    print(f"{'model':<18}{'MAE':>8}{'bias':>9}{'rho':>8}{'slope':>8}")
    print("=" * 74)
    singles = {}
    for k in members:
        s = st([r[k] for r in rows], ref)
        singles[k] = s
        print(f"{k:<18}{s['mae']:>8.2f}{s['bias']:>9.2f}{s['rho']:>8.3f}{s['slope']:>8.3f}")

    print("\n" + "=" * 74)
    print("[Combinations] (equal-weight mean barrier)")
    print(f"{'combination':<26}{'MAE':>8}{'bias':>9}{'rho':>8}{'slope':>8}")
    print("=" * 74)
    combos = []
    for r_ in range(2, len(members) + 1):
        for c in itertools.combinations(members, r_):
            combos.append(c)
    results = {}
    for c in combos:
        s = st([np.mean([row[k] for k in c]) for row in rows], ref)
        results["+".join(c)] = s
        print(f"{'+'.join(c):<26}{s['mae']:>8.2f}{s['bias']:>9.2f}"
              f"{s['rho']:>8.3f}{s['slope']:>8.3f}")

    best = min(singles.items(), key=lambda kv: kv[1]["mae"])
    print(f"\nBest single member: {best[0]} (MAE {best[1]['mae']:.2f})")
    if results:
        bc = min(results.items(), key=lambda kv: kv[1]["mae"])
        print(f"Best combination  : {bc[0]} (MAE {bc[1]['mae']:.2f})")

    # Key question: is the committee better than its best single member?
    full = "+".join(members)
    if full in results:
        fm = results[full]["mae"]
        print(f"\nEqual-weight full committee MAE {fm:.2f} vs best single member {best[1]['mae']:.2f} -> "
              f"{'committee is better' if fm < best[1]['mae'] else 'WARNING: committee is worse than its best single member'}")
        print(f"  (bias: committee {results[full]['bias']:+.2f} "
              f"vs members " + " / ".join(f"{k} {singles[k]['bias']:+.2f}" for k in members) + ")")

    json.dump({"n_valid": len(rows), "singles": singles, "combos": results},
              open(args.inp.replace(".json", "_committee.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"\nWrote {args.inp.replace('.json', '_committee.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
