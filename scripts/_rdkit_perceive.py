"""Helper script: 3D coordinates -> canonical SMILES + independently generated 3D reference.

Run this with a Python interpreter that has RDKit installed (the caller
selects it, e.g. through the JNUSCOUT_RDKIT_PY environment variable).

Two key decisions:

1. Do *not* use the SCINE bond orders: ``su.BondDetector.detect_bonds()``
   returns connectivity only, with every bond order fixed at 1.0 (butadiene
   C=C at 1.34 A also reports 1.0).  Writing a MOL block from that would make
   RDKit read cyclohexene as cyclohexane (``C1CCCCC1``).  RDKit's own
   ``rdDetermineBonds`` (the xyz2mol algorithm) is used instead: it perceives
   connectivity and bond orders from the 3D coordinates together.

2. ``allowChargedFragments=False``: the default behavior generates formal
   charges to satisfy the valence (on open-shell or distorted structures it
   emits fragment ions such as ``[H+]``, ``[C-]``, ``[CH]``, producing
   false-positive deltas of 136-2567 kcal/mol).  Disabling it makes RDKit use
   radical electrons instead, which is correct for the neutral systems this
   tool targets.  Search libraries have been observed to contain radical
   structures with an odd electron count -- this is that pitfall.

Output: ``<out>.smi`` (SMILES; the first stdout line carries the same text)
and ``<out>.xyz`` (3D structure regenerated from that SMILES).
The second stderr line carries diagnostics (formal charges / radical
electrons / fragment count) so the caller can judge whether the perception
is trustworthy.

Usage: <rdkit-python> _rdkit_perceive.py <in.xyz> <out_prefix>
"""
import sys

from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds


def main():
    src, out = sys.argv[1], sys.argv[2]
    raw = Chem.MolFromXYZFile(src)
    if raw is None:
        print("ERR XYZ parsing failed")
        return 1
    mol = Chem.Mol(raw)
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0, allowChargedFragments=False)
    except Exception as e:                                      # noqa: BLE001
        print(f"ERR bond-order perception failed: {type(e).__name__}: {e}")
        return 2

    smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
    # Diagnostics: formal charges / radical electrons -- let the caller judge
    # whether this perception can be trusted.
    n_chg = sum(abs(a.GetFormalCharge()) for a in mol.GetAtoms())
    n_rad = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    n_frag = len(Chem.GetMolFrags(mol))
    print(smi)
    sys.stderr.write(f"DIAG charge={n_chg} radical={n_rad} frags={n_frag}\n")
    with open(out + ".smi", "w") as fh:
        fh.write(smi + "\n")

    # Regenerate a 3D structure from the molecular identity (the original
    # coordinates are not reused).
    m2 = Chem.MolFromSmiles(smi)
    if m2 is None:
        print("ERR SMILES round-trip failed")
        return 3
    m2 = Chem.AddHs(m2)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = 0xC0FFEE
    if AllChem.EmbedMolecule(m2, ps) != 0:
        print("ERR 3D embedding failed")
        return 4
    try:
        AllChem.MMFFOptimizeMolecule(m2, maxIters=5000)
    except Exception:                                           # noqa: BLE001
        pass                                    # radicals sometimes lack MMFF parameters; keep the unoptimized geometry
    conf = m2.GetConformer()
    with open(out + ".xyz", "w") as fh:
        fh.write(f"{m2.GetNumAtoms()}\n{smi}\n")
        for a in m2.GetAtoms():
            p = conf.GetAtomPosition(a.GetIdx())
            fh.write(f"{a.GetSymbol():<2} {p.x:12.6f} {p.y:12.6f} {p.z:12.6f}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
