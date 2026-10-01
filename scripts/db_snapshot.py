"""End-to-end closed-loop database snapshot (refresh the summary numbers).

Prints, for each database: structure count, elementary-step count,
structure count by atom size, and ES count by reaction type
(reactant formula(s) -> product formula(s)).

Usage:
  python scripts/db_snapshot.py --dbs diels_alder_nnp h2co_nnp
"""
import argparse
from collections import Counter

import scine_database as db


def formula(atoms):
    c = Counter(str(atoms[i].element) for i in range(atoms.size()))
    return "".join(f"{k}{v if v > 1 else ''}" for k, v in sorted(c.items()))


def snapshot(m, name):
    ess = m.get_collection("elementary_steps")
    sc = m.get_collection("structures")

    n_struct = sc.count("{}")
    n_es = ess.count("{}")
    print(f"===== {name} =====")
    print(f"structures        {n_struct}")
    print(f"elementary_steps  {n_es}")

    by_size = Counter()
    bad = 0
    for s in sc.iterate_all_structures():
        s.link(sc)
        try:
            by_size[s.get_atoms().size()] += 1
        except Exception:                                      # noqa: BLE001
            bad += 1
    if bad:
        print(f"  (size read failed on {bad} structures)")
    print("structures by atom count:")
    for k in sorted(by_size, key=lambda x: int(x) if str(x).isdigit() else 999):
        print(f"   {k:>4} atoms : {by_size[k]}")

    types = Counter()
    errs = Counter()
    for e in ess.iterate_all_elementary_steps():
        e.link(ess)
        try:
            r, p = e.get_reactants(db.Side.BOTH)
            rk = " + ".join(sorted(formula(db.Structure(db.ID(str(i)), sc).get_atoms()) for i in r))
            pk = " + ".join(sorted(formula(db.Structure(db.ID(str(i)), sc).get_atoms()) for i in p))
            types[f"{rk}  ->  {pk}"] += 1
        except Exception as ex:                                # noqa: BLE001
            errs[type(ex).__name__] += 1
    print("elementary steps by reaction type (top 10):")
    for k, v in types.most_common(10):
        print(f"   {v:>5}  {k}")
    if errs:
        print(f"  (unclassified: {dict(errs)})")

    n16 = by_size.get(16, 0)
    n20 = by_size.get(20, 0)
    da = sum(v for k, v in types.items() if "C2H4" in k.split("->")[0] and "C6H10" in k.split("->")[1])
    print(f"SUMMARY {name}: struct={n_struct} es={n_es} n16={n16} n20={n20} DA(C2H4+C4H6->C6H10)={da}")
    print()
    return n_struct, n_es, n16, n20, da


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbs", nargs="*", default=["diels_alder_nnp", "h2co_nnp"],
                    help="database names to snapshot (default: diels_alder_nnp h2co_nnp)")
    a = ap.parse_args()
    for name in a.dbs:
        cred = db.Credentials()
        cred.hostname, cred.port, cred.database_name = "127.0.0.1", 27017, name
        m = db.Manager()
        m.set_credentials(cred)
        m.connect()
        try:
            snapshot(m, name)
        except Exception as ex:                                # noqa: BLE001
            print(f"===== {name} FAILED: {type(ex).__name__}: {ex}\n")


if __name__ == "__main__":
    main()
