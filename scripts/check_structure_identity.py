"""Independent-source structure cross-check -- a reusable verification test.

Motivation: in a DFTB3 benchmark library used for Diels-Alder work, a
structure had been treated as cyclohexene for many rounds of analysis without
ever being verified.  Generating an independent RDKit reference and comparing
revealed that the library contained no true cyclohexene at all: its lowest
ring C6H10 sat 22.9 kcal/mol above true cyclohexene on the same DFTB3
potential energy surface, while being labeled MINIMUM_OPTIMIZED with
essentially no relaxation movement and perfectly normal-looking bond lengths
(C=C 1.334 A).  Geometry-based criteria alone cannot detect this.

What this tool does (closed self-check loop):

    database structure
      -> write XYZ (coordinates only; bond orders are *not* filled in here)
      -> RDKit (separate interpreter) perceives connectivity and bond orders
         from the 3D coordinates via rdDetermineBonds -> canonical SMILES
      -> regenerate a 3D structure from that SMILES (ETKDGv3 + MMFF)
      -> relax the reference structure on the same potential energy surface
         (DFTB3)
      -> compare energies: a large delta means the library structure is not
         the lowest configuration of that molecule (flagged)

Warning: do *not* use ``su.BondDetector.detect_bonds()`` to fill in bond
orders.  It returns connectivity only, with every bond order fixed at 1.0
(butadiene C=C at 1.34 A also reports 1.0) -- writing a MOL block from that
makes RDKit read cyclohexene as cyclohexane (``C1CCCCC1``).  Bond orders must
be perceived by RDKit itself.

Criterion: delta = E(database structure) - E(independent RDKit reference,
after relaxation).  delta ~ 0 means the structure is fine; a clearly positive
delta means the library entry is wrong (or at least is not the lowest
configuration of that molecule).

Limits:
  * only checks whether the structure is the lowest configuration for its
    connectivity; it cannot tell whether the connectivity itself is
    chemically reasonable;
  * the reference comes from an MMFF force field plus DFTB3 relaxation and
    may land in a different conformational local minimum than the database
    structure -> reliable for small (rigid) molecules, small false-positive
    rate for flexible chains; interpret the magnitude of delta with care;
  * DFTB3 is used as the evaluation surface (fast, native to the stack).
    Other surfaces require an extension.

Statistics must be grouped by label.  The vast majority of entries returned
by ``iterate_all_structures`` are transition states -- ``TS_*`` should not
sit at minima by definition, so a large delta there is expected, not a
defect.  Percentages computed without grouping are meaningless.
  -> the default is ``--label minima``: only structures that *claim* to be
     relaxed minima are considered.

RDKit is reached through a helper script run by a separate Python
interpreter: set ``JNUSCOUT_RDKIT_PY`` to an interpreter with RDKit
installed.  The helper defaults to ``_rdkit_perceive.py`` next to this
script (override with ``JNUSCOUT_RDKIT_HELPER``).

Usage:
    python scripts/check_structure_identity.py --db diels_alder --label minima --sample 60
    python scripts/check_structure_identity.py --db diels_alder --label all --sample 60
    python scripts/check_structure_identity.py --db diels_alder --id 980bb4c8   # single entry
"""
import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter

import numpy as np
import scine_database as db
import scine_utilities as su
import scine_sparrow                                             # noqa: F401
import scine_readuct as readuct

BOHR = 1.8897259886
# The name carries the direction: *multiply* by KCAL_PER_HA to obtain kcal.
# Do not define HA_PER_KCAL ("how many Ha per kcal") and then rely on memory
# to decide whether to multiply or divide.
KCAL_PER_HA = 627.509474
# RDKit is reached through a separate interpreter (see the module docstring).
RDKIT_PY = os.environ.get("JNUSCOUT_RDKIT_PY")
# The perception helper ships next to this script; override for a custom copy.
HELPER = os.environ.get("JNUSCOUT_RDKIT_HELPER", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_rdkit_perceive.py"))
Z = {1: "H", 6: "C", 8: "O", 7: "N", 9: "F", 16: "S", 17: "Cl"}
Z_INV = {v: k for k, v in Z.items()}

# ---- Label groups ----------------------------------------------------------
# Filtering by label is mandatory: the vast majority of structures in the
# library are transition states, and transition states should not sit at
# minima by definition -- a large delta there is expected, not a defect.
SADDLE = {"TS_GUESS", "TS_OPTIMIZED"}
# "Claims to be a relaxed minimum" -- only in this class does a large delta
# count as a defect.
MINIMA_OPTIMIZED = {"MINIMUM_OPTIMIZED", "USER_OPTIMIZED", "SURFACE_OPTIMIZED",
                    "COMPLEX_OPTIMIZED", "REACTIVE_COMPLEX_OPTIMIZED",
                    "USER_SURFACE_OPTIMIZED", "USER_COMPLEX_OPTIMIZED",
                    "SURFACE_COMPLEX_OPTIMIZED", "USER_SURFACE_COMPLEX_OPTIMIZED"}
# "Guessed minimum" -- not relaxed; a large delta only says the initial guess
# is poor, which is a different matter from landing in the wrong basin.
MINIMA_GUESS = {"MINIMUM_GUESS", "USER_GUESS", "COMPLEX_GUESS", "SURFACE_GUESS",
                "REACTIVE_COMPLEX_GUESS", "REACTIVE_COMPLEX_SCANNED",
                "ELEMENTARY_STEP_GUESS", "ELEMENTARY_STEP_OPTIMIZED",
                "SURFACE_ADSORPTION_GUESS"}
NOISY = {"DUPLICATE", "IRRELEVANT", "NONE"}
PRESETS = {"minima": MINIMA_OPTIMIZED, "guess": MINIMA_GUESS,
           "saddle": SADDLE, "all": None}


def _require_rdkit_tools():
    """Fail with a clear message when the RDKit environment is not configured."""
    if not RDKIT_PY:
        raise SystemExit(
            "error: no RDKit interpreter configured; set JNUSCOUT_RDKIT_PY to "
            "a Python executable with RDKit installed")
    if not HELPER or not os.path.exists(HELPER):
        raise SystemExit(
            f"error: RDKit perception helper not found at {HELPER!r}; place "
            "_rdkit_perceive.py next to this script or set JNUSCOUT_RDKIT_HELPER")


def label_name(s):
    """Short name of a structure label (``db.Label.TS_GUESS`` -> ``TS_GUESS``)."""
    try:
        return str(s.get_label()).rsplit(".", 1)[-1]
    except Exception:                                            # noqa: BLE001
        return "?"

_calc = su.core.get_calculator("DFTB3")
_calc.settings["max_scf_iterations"] = 200


def E_single(atoms, mult=1):
    q = su.core.get_calculator("DFTB3")
    q.settings["max_scf_iterations"] = 200
    q.settings["spin_multiplicity"] = mult
    q.structure = atoms
    pl = su.PropertyList()
    pl.add_property(su.Property.Energy)
    q.set_required_properties(pl)
    return float(q.calculate().energy)


def E_relax(atoms, mult=1, max_iter=400):
    q = su.core.get_calculator("DFTB3")
    q.settings["max_scf_iterations"] = 200
    q.settings["spin_multiplicity"] = mult
    s = {"s": q}
    s["s"].structure = atoms
    out, ok = readuct.run_opt_task(s, ["s"], optimizer="bfgs",
                                   convergence_max_iterations=max_iter, output="")
    r = out["s"].structure
    return E_single(r, mult), r


def multiplicity(atoms):
    nel = int(sum(int(atoms[i].element) for i in range(atoms.size())))
    return 1 if nel % 2 == 0 else 2


def to_xyz(atoms, title="db"):
    """Return an XYZ block (Angstrom).

    Bond orders are deliberately *not* written here:
    ``su.BondDetector.detect_bonds()`` returns connectivity only, with every
    bond order fixed at 1.0 (butadiene C=C at 1.34 A also reports 1.0), so
    writing a MOL block from it would make RDKit read cyclohexene as
    cyclohexane.  Bond orders are left to RDKit's ``rdDetermineBonds``,
    perceived from the 3D coordinates.
    """
    pos = np.array(atoms.positions, dtype=float) / BOHR          # Bohr -> Angstrom
    lines = [f"{atoms.size()}", title]
    for i in range(atoms.size()):
        s = Z.get(int(atoms[i].element), "C")
        lines.append(f"{s:<2} {pos[i][0]:12.6f} {pos[i][1]:12.6f} {pos[i][2]:12.6f}")
    return "\n".join(lines) + "\n"


def rdkit_reference(atoms, workdir):
    """Perceive the molecular identity: -> (SMILES, reference AtomCollection or None, diagnostics).

    Runs the RDKit helper in a subprocess.
    """
    pre = os.path.join(workdir, "s")
    with open(pre + ".xyz", "w") as fh:
        fh.write(to_xyz(atoms))
    try:
        r = subprocess.run([RDKIT_PY, HELPER, pre + ".xyz", pre],
                           capture_output=True, text=True, timeout=300)
    except Exception:                                           # noqa: BLE001
        return None, None
    if r.returncode != 0 or not os.path.exists(pre + ".xyz"):
        return None, None, None
    smi = r.stdout.strip().splitlines()[0] if r.stdout.strip() else None
    # Parse the diagnostic line (DIAG charge=.. radical=.. frags=..) -- used
    # to decide whether this perception is trustworthy.
    diag = {}
    for ln in (r.stderr or "").splitlines():
        if ln.startswith("DIAG "):
            for kv in ln[5:].split():
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    diag[k] = int(v)
    els, pos = [], []
    with open(pre + ".xyz") as fh:
        for ln in fh.read().splitlines()[2:]:
            if ln.strip():
                p = ln.split()
                els.append(Z_INV[p[0]])          # ElementType takes no symbol; use atomic numbers
                pos.append([float(x) for x in p[1:4]])
    if len(els) != atoms.size():
        return smi, None, diag              # atom count changed = perception failed; reject the reference
    # AtomCollection wants list[ElementType], not list[int].
    return (smi, su.AtomCollection([su.ElementType(z) for z in els],
                                   np.array(pos, dtype=float) / 0.5291772109), diag)


@contextlib.contextmanager
def _quiet():
    """Silence the SCINE C++ layer's progress log.

    It writes **directly to fd 1** (the ``CONVERGED AFTER N ITERATIONS``
    block), which Python's ``contextlib.redirect_stdout`` cannot intercept --
    the file descriptor itself must be swapped.
    (``readuct.run_opt_task(output="")`` only silences readuct's own layer.)
    """
    sys.stdout.flush()
    with open(os.devnull, "w") as dn:
        old = os.dup(1)
        os.dup2(dn.fileno(), 1)
        try:
            yield
        finally:
            sys.stdout.flush()
            os.dup2(old, 1)
            os.close(old)


def check(atoms, workdir, trust_only=True):
    with _quiet():
        return _check(atoms, workdir, trust_only)


def _check(atoms, workdir, trust_only=True):
    """delta = E(database structure, as-is) - E(independent RDKit reference, relaxed).

    Identity perception uses the *relaxed* geometry: on a distorted geometry
    any perception has to invent either formal charges or radicals (the
    C4H6-as-``[CH]C[CH][CH2]`` four-radical artifact comes from this).
    Relaxing to the true minimum on this potential energy surface first
    yields a clean Lewis structure, while delta is still evaluated with the
    original structure's energy, so the diagnostic power ("how far is this
    structure from its own minimum") is preserved.
    """
    m = multiplicity(atoms)
    e_db = E_single(atoms, m)
    e_opt, at_opt = E_relax(atoms, m)        # relaxation only to establish identity
    # Divergence guard: readuct.run_opt_task can diverge silently -- observed
    # geometries stretched to |P|max 8.7 A with all C-C bonds broken, while the
    # return value and ok still look normal.  Without this guard the diverged
    # geometry would both be sent to perception (yielding bogus SMILES) and
    # produce absurd deltas (a previous version of this tool reported
    # +2567 kcal/mol this way).
    _p0 = np.abs(np.array(atoms.positions, dtype=float)).max()
    _p1 = np.abs(np.array(at_opt.positions, dtype=float)).max()
    if _p1 > 3.0 * max(_p0, 1.0):
        return {"ok": False,
                "why": f"relaxation diverged (|P|max {_p0:.1f} -> {_p1:.1f} Bohr)"}
    smi, ref, diag = rdkit_reference(at_opt, workdir)
    if ref is None:
        return {"smiles": smi, "ok": False, "diag": diag,
                "why": "RDKit reference generation failed / atom count mismatch"}
    # Trust gate: a formal charge means RDKit is forcing the valence bonds, so
    # the perception cannot be trusted.
    if trust_only and diag and diag.get("charge", 0) > 0:
        return {"smiles": smi, "ok": False, "diag": diag,
                "why": f"perception untrusted (formal charge {diag.get('charge')})"}
    if trust_only and diag and diag.get("frags", 1) > 1:
        return {"smiles": smi, "ok": False, "diag": diag,
                "why": f"perception untrusted (split into {diag.get('frags')} fragments)"}
    # Radical gate: if the library structure is even-electron (should be a
    # closed-shell singlet) and RDKit *still* perceives radicals after
    # relaxation, the relaxation itself has changed the chemical species
    # (stretched bonds force xyz2mol to use radicals to satisfy the valence).
    # The reference is then a different species and delta is meaningless --
    # it would produce both false "wrong basin" (69.80 / 69.55) and false
    # "healthy" (-3.99) results.  ``allowChargedFragments=False`` only trades
    # the charge artifact for a radical artifact; this gate is the real
    # closure.
    if (trust_only and m == 1 and diag and diag.get("radical", 0) > 0):
        return {"smiles": smi, "ok": False, "diag": diag,
                "why": f"perception untrusted (even-electron structure read as "
                       f"{diag.get('radical')} radicals; relaxation changed the chemical species)"}
    e_ref, ref_min = E_relax(ref, m)
    d = (e_db - e_ref) * KCAL_PER_HA
    # Extra diagnostic: how far the database structure itself ends up from
    # the reference after its own relaxation.
    #   ~0 -> it was already in the correct basin (just not fully relaxed)
    #   still large -> it relaxed into another, higher basin (a true wrong basin)
    d_opt = (e_opt - e_ref) * KCAL_PER_HA
    return {"smiles": smi, "ok": True, "e_db": e_db, "e_ref": e_ref,
            "d_kcal": float(d), "d_opt_kcal": float(d_opt), "mult": m, "diag": diag}


def verdict(r, flag=5.0):
    """Separate the two failure modes with two columns (delta, delta after relaxation).

    | delta | delta(relaxed) | meaning |
    |---|---|---|
    | small | small | healthy |
    | large | ~0 | not converged -- was never relaxed to the minimum when stored |
    | large | large | wrong basin -- relaxed to a different (wrong) local minimum |

    This reading is only valid when the label belongs to the class that
    *should* be at a minimum; grouping is the caller's responsibility.
    """
    if not r.get("ok"):
        return "untrusted"
    d = r["d_kcal"]
    do = r.get("d_opt_kcal", d)
    if d < flag:
        return "healthy" if do < flag else "worse after relaxation"
    return "not converged" if do < flag else "wrong basin"


def selftest():
    """Negative control: run known-correct molecules through the full pipeline.

    No database involved -- build a structure from SMILES, then run check().
    It tests whether the tool itself reports spuriously; if delta is not ~0,
    refuse to produce a report.
    """
    refs = []
    for smi in ("C1CCC=CC1", "C=CC=C"):
        with tempfile.TemporaryDirectory() as wd:
            f = os.path.join(wd, "ref")
            # Write a minimal RDKit generation script to a file to avoid nested
            # escaping in the command line.
            with open(f + ".py", "w") as fh:
                fh.write(
                    "import sys\n"
                    "from rdkit import Chem\n"
                    "from rdkit.Chem import AllChem\n"
                    "m = Chem.AddHs(Chem.MolFromSmiles(sys.argv[1]))\n"
                    "ps = AllChem.ETKDGv3(); ps.randomSeed = 1\n"
                    "AllChem.EmbedMolecule(m, ps)\n"
                    "AllChem.MMFFOptimizeMolecule(m, maxIters=2000)\n"
                    "c = m.GetConformer()\n"
                    "print(len(m.GetAtoms()))\n"
                    "print(sys.argv[1])\n"
                    "for a in m.GetAtoms():\n"
                    "    p = c.GetAtomPosition(a.GetIdx())\n"
                    "    print(a.GetSymbol(), p.x, p.y, p.z)\n")
            r = subprocess.run([RDKIT_PY, f + ".py", smi],
                               capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                continue
            rows = r.stdout.splitlines()
            els, pos = [], []
            for row in rows[2:]:
                if row.strip():
                    q = row.split()
                    els.append(su.ElementType(Z_INV[q[0]]))
                    pos.append([float(x) for x in q[1:4]])
        if not els:
            continue
        refs.append((smi, su.AtomCollection(els, np.array(pos, dtype=float) / 0.5291772109)))

    if not refs:
        print("negative control could not be built; skipping")
        return True
    print("Negative control (known-correct molecules, delta should be ~0):")
    okall = True
    for smi, at in refs:
        with tempfile.TemporaryDirectory() as wd:
            r = check(at, wd)
        if not r.get("ok"):
            print("  FAIL %-12s %s" % (smi, r.get("why")))
            okall = False
            continue
        good = abs(r["d_kcal"]) < 2.0
        okall = okall and good
        print("  %s %-12s delta = %+7.2f kcal/mol"
              % ("OK  " if good else "FAIL", smi, r["d_kcal"]))
    return okall


def main():
    ap = argparse.ArgumentParser(
        epilog="RDKit runs in a separate interpreter: set JNUSCOUT_RDKIT_PY to "
               "a Python executable with RDKit installed; the perception helper "
               "defaults to _rdkit_perceive.py next to this script and can be "
               "overridden with JNUSCOUT_RDKIT_HELPER.")
    ap.add_argument("--db", default="diels_alder")
    ap.add_argument("--id", default=None, help="check a single entry only (last 8 chars of the structure id)")
    ap.add_argument("--sample", type=int, default=40, help="number of unique structures to sample")
    ap.add_argument("--flag", type=float, default=5.0, help="flag above this value (kcal/mol)")
    ap.add_argument("--label", default="minima",
                    help="label whitelist, comma-separated, or a preset minima/guess/saddle/all (default: minima)")
    ap.add_argument("--size", type=int, default=None, help="only check structures with this atom count")
    ap.add_argument("--group", action="store_true",
                    help="summarize per label (the summary is grouped by default)")
    ap.add_argument("--out", default="results/identity_check.json")
    a = ap.parse_args()

    _require_rdkit_tools()

    # By default only structures that *claim* to be minima are checked -- without
    # this, the sample would be swamped by transition states (see the label notes
    # in the module docstring).
    if a.label in PRESETS:
        wanted = PRESETS[a.label]
        print(f"label preset '{a.label}' -> "
              f"{'all' if wanted is None else sorted(wanted)}")
    else:
        wanted = {x.strip().upper() for x in a.label.split(",") if x.strip()}
        print(f"label whitelist -> {sorted(wanted)}")

    if not selftest():
        print("\nNegative control failed -- the tool itself is suspect; refusing to report.")
        return

    c = db.Credentials()
    c.hostname, c.port, c.database_name = "127.0.0.1", 27017, a.db
    m = db.Manager()
    m.set_credentials(c)
    m.connect()
    st = m.get_collection("structures")

    seen, pool = set(), []
    for s in st.iterate_all_structures():
        s.link(st)
        sid = str(s.get_id())
        if a.id:
            if not sid.endswith(a.id):
                continue
        else:
            lb = label_name(s)
            if wanted is not None and lb not in wanted:
                continue
        try:
            at = s.get_atoms()
        except Exception:                                       # noqa: BLE001
            continue
        if at.size() == 0:
            continue
        if a.size is not None and at.size() != a.size:
            continue
        key = (at.size(), tuple(np.round(np.array(at.positions, dtype=float).ravel(),
                                         2).tolist()))
        if key in seen:
            continue
        seen.add(key)
        pool.append((sid, at, label_name(s)))

    # Uniform deterministic sampling (no RNG): equally spaced indices over
    # insertion order.  (Taking the first N entries would only cover the
    # earliest-inserted structures, which is not a sample.)
    if a.id or len(pool) <= a.sample:
        todo = pool
    else:
        idx = sorted({int(round(x)) for x in np.linspace(0, len(pool) - 1, a.sample)})
        todo = [pool[i] for i in idx]

    print(f"{a.db}: {len(pool)} unique structures matched -> sampling {len(todo)} "
          f"(threshold {a.flag} kcal/mol)\n")
    print(f"{'structure':<10}{'Label':<20}{'SMILES':<20}{'delta':>8}{'d_relaxed':>11}  verdict")
    print("  delta          = E(DB as-is) - E(reference)   -> how far this structure is from its own minimum")
    print("  d_relaxed      = E(DB relaxed) - E(reference)  "
          "-> ~0 means it was already in the right basin; still large = true wrong basin")
    print("  Note: read the Label first -- TS_* should not sit at a minimum; a large delta is not a defect")
    print()
    res, flagged = [], []
    excl = Counter()
    for sid, at, lb in todo:
        with tempfile.TemporaryDirectory() as wd:
            try:
                r = check(at, wd)
            except Exception as ex:                             # noqa: BLE001
                excl[f"exception {type(ex).__name__}"] += 1
                print(f"{sid[-8:]:<10}{lb:<20}x {type(ex).__name__}: {str(ex)[:50]}")
                continue
        r["id"] = sid[-8:]
        r["size"] = at.size()
        r["label"] = lb
        r["verdict"] = verdict(r, a.flag) if r.get("ok") else "untrusted"
        res.append(r)
        if not r.get("ok"):
            excl[str(r.get("why", "?")).split("(")[0]] += 1
            print(f"{sid[-8:]:<10}{lb:<20}{'-':<20}{'':>8}{'':>11}  x {r.get('why')}")
            continue
        bad = r["verdict"] in ("wrong basin", "not converged")
        if bad:
            flagged.append(r)
        print(f"{sid[-8:]:<10}{lb:<20}{(r['smiles'] or '')[:19]:<20}"
              f"{r['d_kcal']:>8.2f}{r.get('d_opt_kcal', float('nan')):>11.2f}"
              f"  {'!! ' if bad else '   '}{r['verdict']}")

    ok = [r for r in res if r.get("ok")]
    print(f"\n{'=' * 86}\nGrouped by label (the only usable statistics convention)\n{'=' * 86}")
    print(f"{'Label':<24}{'count':>6}{'trusted':>6}{'median d':>10}"
          f"{'wrong basin':>11}{'not conv.':>10}   should be at a minimum?")
    by = {}
    for r in res:
        by.setdefault(r["label"], []).append(r)
    warn = []
    for lb, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        okk = [r for r in rs if r.get("ok")]
        if not okk:
            print(f"{lb:<24}{len(rs):>6}{0:>6}{'-':>10}{'-':>11}{'-':>10}"
                  f"   (all untrusted)")
            continue
        med = float(np.median([r["d_kcal"] for r in okk]))
        wb = [r for r in okk if r["verdict"] == "wrong basin"]
        nc = [r for r in okk if r["verdict"] == "not converged"]
        if lb in MINIMA_OPTIMIZED:
            tag = "yes (large delta = defect)"
        elif lb in MINIMA_GUESS:
            tag = "no (unrelaxed guess)"
        elif lb in SADDLE:
            tag = "no (saddle point)"
        elif lb in NOISY:
            tag = "no (noise label)"
        else:
            tag = "?"
        print(f"{lb:<24}{len(rs):>6}{len(okk):>6}{med:>10.2f}"
              f"{len(wb):>11}{len(nc):>10}   {tag}")
        if lb in MINIMA_OPTIMIZED and wb:
            warn.append((lb, len(wb), len(okk)))
    print()
    print("Only a large delta within the 'should be at a minimum' group is a defect; "
          "a large delta in the saddle-point group is expected.")
    if not warn:
        print("-> No wrong-basin structures among the claimed relaxed minima.")
    else:
        for lb, n, tot in warn:
            print(f"-> {lb}: {n}/{tot} ({100 * n / tot:.0f}%) wrong basin "
                  f"(still not at the correct minimum after relaxation)")

    if ok:
        v = np.array([r["d_kcal"] for r in ok])
        print(f"\nAll samples (informational only -- labels are mixed): "
              f"{len(ok)}/{len(res)} trusted, median delta {np.median(v):+.2f}, "
              f"range [{v.min():+.2f}, {v.max():+.2f}], "
              f"{len(flagged)}/{len(ok)} above threshold")
    if excl:
        print(f"\nExcluded {sum(excl.values())}/{len(todo)} entries "
              f"(untrusted -- must be dropped from the statistics, "
              f"otherwise the numbers are distorted in both directions):")
        for k, v in excl.most_common():
            print(f"  {k:<30}{v:>5}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"params": vars(a), "excluded": dict(excl), "rows": res},
                  fh, indent=2, ensure_ascii=False, default=str)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
