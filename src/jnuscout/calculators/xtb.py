"""xtb ASE calculator (GFN2-xTB, L1 screening).

Wraps the xtb executable (conda-forge build, compiled with gfortran). ASE 3.29
ships no built-in xtb interface, and the SCINE xtb_wrapper is not usable on the
target host (EPYC Zen 2 CPU without AVX512, plus an ABI conflict), so this
module provides a CLI subprocess wrapper in the same style as ORCACalculator.

Units: energy Eh -> eV; gradient Eh/bohr -> eV/A
(F = -grad * 27.211386245 / 0.5291772109).
"""
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

XTB_BIN = os.environ.get("XTB_BIN", "xtb")
HA2EVA = 27.211386245 / 0.5291772109


class XTBCalculator(Calculator):
    """ASE calculator wrapping the xtb CLI (GFN2-xTB).

    Parameters
    ----------
    method : str
        "GFN2-xTB" (default), "GFN1-xTB", "GFN0-xTB", "GFNFF".
    charge : int
    mult : int
    keep_files : bool
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, method="GFN2-xTB", charge=0, mult=1, keep_files=False,
                 workdir=None, xtb_bin=None):
        super().__init__()
        self.method = method
        self.charge = charge
        self.mult = mult
        self.keep_files = keep_files
        self.workdir = workdir
        self.xtb_bin = xtb_bin or XTB_BIN

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n = len(atoms)
        gfn = {"GFN2-xTB": "2", "GFN1-xTB": "1", "GFN0-xTB": "0", "GFNFF": "ff"}[self.method]
        uhf = self.mult - 1
        if self.keep_files and self.workdir:
            os.makedirs(self.workdir, exist_ok=True)
            tmp = self.workdir
        else:
            tmp = tempfile.mkdtemp(prefix="xtb_")
        xyz_path = os.path.join(tmp, "mol.xyz")
        with open(xyz_path, "w") as f:
            f.write(f"{n}\nxtb\n")
            for s, p in zip(atoms.get_chemical_symbols(), atoms.positions):
                f.write(f"  {s}  {p[0]:.8f}  {p[1]:.8f}  {p[2]:.8f}\n")
        out = os.path.join(tmp, "xtb.out")
        grad = os.path.join(tmp, "gradient")
        try:
            r = subprocess.run(
                [self.xtb_bin, xyz_path, "--gfn", gfn, "--chrg", str(self.charge),
                 "--uhf", str(uhf), "--grad"],
                stdout=open(out, "w"), stderr=subprocess.STDOUT, cwd=tmp)
            if r.returncode != 0 or not os.path.exists(grad):
                raise RuntimeError(f"xtb failed (rc={r.returncode}); see {out}")
            txt = open(out, encoding="utf-8", errors="replace").read()
            m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", txt)
            if not m:
                raise RuntimeError(f"xtb energy not found in {out}")
            energy = float(m.group(1))            # Eh
            # gradient file: $grad header + cycle line + N coordinate lines
            # (4 columns, last column = element symbol) + N gradient lines (3 columns)
            lines = open(grad, encoding="utf-8", errors="replace").read().strip().split("\n")[2:]
            vals = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        vals.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        continue
            # vals: the first N rows are coordinates, the last N rows are the
            # gradients (Eh/bohr)
            gradient = np.array(vals[n:2 * n]).reshape(n, 3)
            if gradient.shape != (n, 3):
                raise RuntimeError(f"gradient shape mismatch: {gradient.shape}")
        finally:
            if not self.keep_files:
                shutil.rmtree(tmp, ignore_errors=True)
        self.results = {
            "energy": energy * 27.211386245,      # Eh -> eV
            "forces": -gradient * HA2EVA,          # Eh/bohr -> eV/A
        }
