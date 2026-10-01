"""ORCA ASE calculator for JNUScout.

Layer definitions:
  L2: r2SCAN-D4/def2-SVP, RIJCOSX approximation with the def2/J auxiliary
      basis, TightSCF.  r2SCAN is a pure mGGA without HF exchange, so ORCA
      automatically downgrades RIJCOSX to Split-RI-J (a harmless warning).
  L3: wB97M-V/def2-TZVP, RIJCOSX with an AutoAux-generated auxiliary basis,
      defgrid3, TightSCF.

L1 (xTB) is not part of this module -- xTB is a standalone program and is
handled separately when integrating with Chemoton.

Units: ASE energy = eV; forces = eV/Angstrom.
ORCA engrad output: energy in Eh, gradient in Eh/bohr;
conversion F[eV/A] = -grad * 27.211386245 / 0.5291772109.
"""
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

ORCA_BIN = os.environ.get("ORCA_BIN", "orca")
HA2EVA = 27.211386245 / 0.5291772109  # Eh/bohr -> eV/A

ORCA_KEYWORDS = {
    # For r2SCAN (a pure mGGA), RIJCOSX is automatically downgraded by ORCA to
    # Split-RI-J (a harmless warning) -- ORCA has no "RI-J" keyword
    # (confirmed by testing).
    "L2": "r2SCAN D4 def2-SVP RIJCOSX def2/J TightSCF",
    # The originally specified def2/JK auxiliary basis was unavailable on the
    # deployment host (no local auxiliary basis file and no network access for
    # automatic download), so AutoAux is used instead (confirmed by testing).
    "L3": "wB97M-V def2-TZVP RIJCOSX AutoAux defgrid3 TightSCF",
}


def parse_engrad(path):
    """Parse ORCA .engrad file.

    Returns (energy_Eh, gradient_Eh_per_bohr, coords_bohr).
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        txt = f.read()
    energy = float(txt.split("The current total energy in Eh")[1].split("#")[1].strip())
    g_sec = txt.split("The current gradient in Eh/bohr")[1].split("# The atomic numbers")[0]
    grad = np.array([float(t) for t in re.findall(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", g_sec)])
    n = grad.size // 3
    grad = grad.reshape(n, 3)
    c_sec = txt.split("The atomic numbers and current coordinates in Bohr")[1]
    coords = np.array([float(t) for t in re.findall(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", c_sec)])
    # Atomic numbers are integers (no decimal point) and are not matched by the
    # regex above; if they appear in decimal form (4n entries), skip the first n.
    if coords.size == 4 * n:
        coords = coords[n:]
    coords = coords[: 3 * n].reshape(n, 3)
    return energy, grad, coords


# Common locations of the OpenMPI runtime libraries (ORCA's _shared_openmpi
# distribution requires libmpi.so).
_EXTRA_LIB_CANDIDATES = [
    "/usr/local/openmpi-4.1.8/lib",
    "/usr/lib64/openmpi/lib",
    "/usr/lib/x86_64-linux-gnu/openmpi/lib",
]


def _orca_env(orca_bin):
    """Build the subprocess environment for ORCA: make sure LD_LIBRARY_PATH
    contains the ORCA directory and the OpenMPI runtime libraries.

    Background: on the deployment host LD_LIBRARY_PATH is set by .bashrc
    (ORCA directory + /usr/local/openmpi-4.1.8/lib). When ORCA is launched via
    systemd-run, a cron job, or a non-interactive shell, .bashrc is not loaded
    (a non-interactive shell returns at the top), so the ORCA binary (a 36 MB
    monolithic executable without a wrapper script) cannot find liborca /
    libmpi and exits immediately -- the symptom is `rc=0 but no .engrad file`,
    which is easily mistaken for an ORCA calculation failure.

    The environment variable ORCA_EXTRA_LIB_DIRS (colon-separated) can be used
    to append additional library directories.
    """
    env = os.environ.copy()

    # 1) LD_LIBRARY_PATH: ORCA itself + the OpenMPI runtime libraries
    paths = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p]
    extra = [os.path.dirname(os.path.abspath(orca_bin))]
    extra += [p for p in env.get("ORCA_EXTRA_LIB_DIRS", "").split(":") if p]
    extra += [p for p in _EXTRA_LIB_CANDIDATES if os.path.isdir(p)]
    for p in reversed(extra):          # keep priority: ORCA directory first
        if p not in paths:
            paths.insert(0, p)
    env["LD_LIBRARY_PATH"] = ":".join(paths)

    # 2) PATH: ORCA's parallel modules invoke mpirun (if it is missing, ORCA
    # reports "mpirun: command not found" -> error termination in Startup)
    bins = [p for p in env.get("ORCA_EXTRA_BIN_DIRS", "").split(":") if p]
    for libdir in _EXTRA_LIB_CANDIDATES:
        bindir = os.path.normpath(os.path.join(libdir, os.pardir, "bin"))
        if os.path.isdir(bindir):
            bins.append(bindir)
    if bins:
        cur = [p for p in env.get("PATH", "").split(":") if p]
        for b in reversed(bins):
            if b not in cur:
                cur.insert(0, b)
        env["PATH"] = ":".join(cur)
    return env


def make_orca_input(atoms, keywords, nprocs=8, maxcore=4000):
    """Generate ORCA input text from an ASE Atoms object."""
    lines = [f"! {keywords} Engrad", f"%pal nprocs {nprocs} end", f"%maxcore {maxcore}"]
    lines.append(f"* xyz {atoms.get_initial_charges().sum() if atoms.has('initial_charges') else 0} {atoms.get_initial_magnetic_moments().sum() if atoms.has('initial_magnetic_moments') else 1}")
    for s, p in zip(atoms.get_chemical_symbols(), atoms.positions):
        lines.append(f"  {s}   {p[0]:.8f}   {p[1]:.8f}   {p[2]:.8f}")
    lines.append("*")
    return "\n".join(lines)


class ORCACalculator(Calculator):
    """ASE calculator wrapping ORCA single-point energy+gradient.

    Parameters
    ----------
    level : str
        "L2" (r2SCAN-D4/def2-SVP) or "L3" (wB97M-V/def2-TZVP).
    nprocs : int
    maxcore : int
    keep_files : bool
        If True, keep the ORCA input/output in `workdir`; else use a temp dir.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, level="L2", nprocs=8, maxcore=4000, keep_files=False,
                 workdir=None, orca_bin=None):
        super().__init__()
        if level not in ORCA_KEYWORDS:
            raise ValueError(f"level must be one of {list(ORCA_KEYWORDS)}")
        self.level = level
        self.keywords = ORCA_KEYWORDS[level]
        self.nprocs = nprocs
        self.maxcore = maxcore
        self.keep_files = keep_files
        self.workdir = workdir
        self.orca_bin = orca_bin or ORCA_BIN

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n = len(atoms)
        inp = make_orca_input(atoms, self.keywords, self.nprocs, self.maxcore)
        if self.keep_files and self.workdir:
            os.makedirs(self.workdir, exist_ok=True)
            base = os.path.join(self.workdir, f"run_{os.getpid()}")
        else:
            tmp = tempfile.mkdtemp(prefix="orca_")
            base = os.path.join(tmp, "run")
        with open(base + ".inp", "w") as f:
            f.write(inp)
        out = base + ".out"
        engrad = base + ".engrad"
        ok = False
        try:
            r = subprocess.run([self.orca_bin, base + ".inp"],
                               stdout=open(out, "w"), stderr=subprocess.STDOUT,
                               cwd=os.path.dirname(base) or ".", env=_orca_env(self.orca_bin))
            if r.returncode != 0 or not os.path.exists(engrad):
                raise RuntimeError(f"ORCA failed for {base}.inp (rc={r.returncode}); see {out}")
            energy, grad, coords = parse_engrad(engrad)
            if len(coords) != n:
                raise RuntimeError(f"engrad atom count mismatch: {len(coords)} vs {n}")
            ok = True
        finally:
            if ok and not self.keep_files:
                shutil.rmtree(tmp, ignore_errors=True)
        self.results = {
            "energy": energy * 27.211386245,          # Eh -> eV
            "forces": -grad * HA2EVA,                  # Eh/bohr -> eV/A, F = -dE/dr
        }
