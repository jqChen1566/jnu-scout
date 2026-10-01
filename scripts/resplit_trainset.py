"""Stratified re-split of the training / validation set by composition.

Problem: an initial split that simply took the tail of the ES order produced a
training set of C4H6 (600 frames) and a validation set of C6H10 (120 frames) --
completely different compositions.  MACE then evaluates the validation set with
the E0s of the training set, giving an energy RMSE of 29 eV/atom and a rising
loss.

Fix: merge both batches and split **stratified by composition** -- the
elementary steps of each composition are split 8:2 into training / validation
so that both sides share the same composition distribution.

Usage:
    python scripts/resplit_trainset.py --a data/al_train_da.xyz --b data/al_holdout_da.xyz \
        --out-train data/al_train2.xyz --out-valid data/al_valid2.xyz
"""
import argparse
from collections import Counter, defaultdict


def read_xyz(path):
    """Read an extxyz file; returns [(header, [(el,x,y,z,fx,fy,fz)], es_id, t, energy)]."""
    frames = []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            n = int(line)
            hdr = f.readline().strip()
            atoms = [f.readline().strip() for _ in range(n)]
            es_id = ""
            t = ""
            e = 0.0
            for token in hdr.split():
                if token.startswith("es_id="):
                    es_id = token.split("=", 1)[1]
                elif token.startswith("t="):
                    t = token.split("=", 1)[1]
                elif token.startswith("energy="):
                    e = float(token.split("=", 1)[1])
            comp = "".join(sorted(Counter(a.split()[0] for a in atoms).elements()))
            frames.append({"n": n, "hdr": hdr, "atoms": atoms, "es_id": es_id,
                           "t": t, "energy": e, "comp": comp,
                           "pos": [tuple(map(float, a.split()[1:4])) for a in atoms]})
    return frames


def same_geometry(f1, f2, tol=1e-6):
    if f1["comp"] != f2["comp"]:
        return False
    for p1, p2 in zip(f1["pos"], f2["pos"]):
        if abs(p1[0] - p2[0]) > tol or abs(p1[1] - p2[1]) > tol or abs(p1[2] - p2[2]) > tol:
            return False
    return True


def write_xyz(path, frames):
    with open(path, "w") as f:
        for fr in frames:
            f.write(f"{fr['n']}\n")
            f.write(fr["hdr"] + "\n")
            for a in fr["atoms"]:
                f.write(a + "\n")
    return len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="data/al_train_da.xyz")
    ap.add_argument("--b", default=None, help="second file (merged in), optional")
    ap.add_argument("--out-train", default="data/al_train2.xyz")
    ap.add_argument("--out-valid", default="data/al_valid2.xyz")
    ap.add_argument("--valid-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260911)
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)

    fa = read_xyz(args.a)
    fb = read_xyz(args.b) if args.b else []
    all_frames = fa + fb
    # Deduplicate (a and b may share overlapping geometries)
    uniq = []
    for fr in all_frames:
        if not any(same_geometry(fr, u) for u in uniq):
            uniq.append(fr)
    print(f"merged {len(fa)} + {len(fb)} = {len(all_frames)} frames, {len(uniq)} after dedup")

    # Stratify by composition
    by_comp = defaultdict(list)
    for fr in uniq:
        by_comp[fr["comp"]].append(fr)

    train, valid = [], []
    for comp, frames in by_comp.items():
        # Group by es_id (all configurations of one path must go to the same side
        # to avoid leakage)
        by_es = defaultdict(list)
        for fr in frames:
            by_es[fr["es_id"]].append(fr)
        es_ids = sorted(by_es)
        rng.shuffle(es_ids)
        n_valid_es = max(1, int(round(len(es_ids) * args.valid_frac)))
        valid_es = set(es_ids[:n_valid_es])
        for eid, frs in by_es.items():
            (valid if eid in valid_es else train).extend(frs)
        print(f"  composition {comp}: {len(frames)} frames / {len(es_ids)} ES -> "
              f"train {sum(len(v) for k, v in by_es.items() if k not in valid_es)} frames, "
              f"valid {sum(len(v) for k, v in by_es.items() if k in valid_es)} frames")

    nt = write_xyz(args.out_train, train)
    nv = write_xyz(args.out_valid, valid)
    print(f"\nWrote {args.out_train}: {nt} frames | {args.out_valid}: {nv} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
