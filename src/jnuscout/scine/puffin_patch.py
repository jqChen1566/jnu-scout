"""Run SCINE tasks in Chemoton/Puffin through an ML-potential bridge.

## Why these functions are intercepted

Every readuct-type task in Puffin (``AFIR = "run_afir_task"``,
``OPT/RCOPT/IRCOPT/TSOPT``, ...) resolves its calculator through a single
choke point:

    scine_puffin/utilities/scine_helper.py:202   SettingsManager.prepare_readuct_task
        utils.io.write("system.xyz", structure.get_atoms())
        system = utils.core.load_system_into_calculator(
            "system.xyz", model.method_family, **self.calculator_settings)

Puffin never receives a calculator object; it resolves one from the
``method_family`` string alone. The correct way to plug a bridge in is
therefore to replace, in-process, the resolution functions on
``scine_utilities.core``: when one of the declared method families is matched,
the bridge constructs the calculator; every other family (DFTB3/PM6/DFT, ...)
falls back to the original implementation untouched.

``SettingsManager.__init__`` additionally calls
``get_available_settings(method_family, program)`` (C++ implementation:
``getCalculator(...)->settings()``); leaving it unpatched makes the
SettingsManager constructor fail before anything else can run.

## Injection strategy (without modifying Puffin sources)

Place the package on ``PYTHONPATH`` and have ``sitecustomize.py`` call
``install_from_env()``. Environment variables (names kept for compatibility
with existing deployments):

    NNP_AFIR_PUFFIN=1              enable the patch
    NNP_AFIR_METHOD_FAMILY=nnp     method family to intercept (comma-separated)
    NNP_AFIR_BACKEND=mace          emt | mace | aimnet2 | committee
    NNP_AFIR_MODEL=/path/x.model   path to the MACE model
    NNP_AFIR_DEVICE=cuda
    NNP_AFIR_PUFFIN_VERBOSE=1      print hit messages

## One pitfall that must be handled (carried over from ``inject.py``)

``SettingsManager.calculator_settings`` contains a ``method_family`` key, so
calling ``load_system_into_calculator(xyz, method_family, **settings)`` would
pass the same argument both positionally and as a keyword. The wrapper must
accept this leniently and de-duplicate.
"""
from __future__ import annotations

import os

import scine_utilities as su

from .bridge import ASECalculatorBridge, bridge_descriptors

DEFAULT_METHOD_FAMILY = "nnp"

_VERBOSE = os.environ.get("NNP_AFIR_PUFFIN_VERBOSE", "") not in ("", "0")

# Installation state
_state = {"installed": False, "families": (), "factory": None,
          "originals": {}, "skipped_settings": {}}


def _log(msg: str) -> None:
    if _VERBOSE:
        print(f"[nnp-puffin] {msg}", flush=True)


# ============================ Backend factory ============================

def make_bridge_from_env() -> ASECalculatorBridge:
    """Create a new bridge from the environment variables."""
    backend = os.environ.get("NNP_AFIR_BACKEND", "mace").lower()
    device = os.environ.get("NNP_AFIR_DEVICE", "cuda")

    if backend == "emt":
        from ase.calculators.emt import EMT
        return ASECalculatorBridge(EMT(), name="nnp_emt")

    if backend == "mace":
        path = os.environ.get("NNP_AFIR_MODEL")
        if not path:
            raise RuntimeError("NNP_AFIR_MODEL must be set when NNP_AFIR_BACKEND=mace")
        from mace.calculators import MACECalculator
        calc = MACECalculator(model_paths=[path], device=device, default_dtype="float64")
        return ASECalculatorBridge(calc, name=f"nnp_mace:{os.path.basename(path)}")

    if backend == "aimnet2":
        # Lazy import: the aimnet2 stack is heavy and its availability varies
        # across machines.
        model = os.environ.get("NNP_AFIR_MODEL", "wb97m_0")
        calc = su.core.get_calculator("AIMNet2")   # placeholder; the real route is the ASE interface
        raise NotImplementedError(f"backend=aimnet2 is not wired up yet (model={model})")

    raise ValueError(f"unknown NNP_AFIR_BACKEND: {backend!r}")


# ============================ Utilities ============================

def _read_structure(first):
    """Convert the given object into an ``su.AtomCollection``.

    This matches SCINE's native semantics:
      * an ``AtomCollection`` is used as-is, with no unit conversion
        (SCINE works in Bohr throughout);
      * a file path is read from disk and converted to Bohr.
    """
    if isinstance(first, su.AtomCollection):
        return first
    path = str(first)
    try:
        read = su.io.read(path)
        return read[0] if isinstance(read, tuple) else read
    except Exception:                                      # noqa: BLE001
        # Fallback: read the xyz file with ASE (Angstrom) and convert to Bohr.
        from ase.io import read as ase_read
        import numpy as np
        from ase.data import atomic_numbers
        at = ase_read(path)
        els = [su.ElementType(int(atomic_numbers[s])) for s in at.get_chemical_symbols()]
        return su.AtomCollection(els, np.asarray(at.get_positions(), float) * 1.8897259886)


def _apply_settings(calc: ASECalculatorBridge, kwargs: dict) -> None:
    """Write the settings kwargs sent by Puffin into the bridge.

    Unknown keys are skipped and counted.
    """
    for k, v in kwargs.items():
        if k == "method_family":
            continue
        try:
            calc.settings[k] = v
        except Exception:                                  # noqa: BLE001
            n = _state["skipped_settings"]
            n[k] = n.get(k, 0) + 1
    if _VERBOSE and _state["skipped_settings"]:
        _log(f"skipped setting keys (no descriptor declared by the bridge): {_state['skipped_settings']}")


def nnp_settings(program: str = "Any") -> "su.Settings":
    """Return value of ``get_available_settings`` -- equivalent to
    ``getCalculator(...)->settings()``."""
    s = su.Settings(bridge_descriptors())
    try:
        s["program"] = program
    except Exception:                                      # noqa: BLE001
        pass
    return s


# ============================ Installation ============================

def install(method_families=DEFAULT_METHOD_FAMILY, factory=None, force: bool = False):
    """Replace the resolution functions on ``scine_utilities.core`` (idempotent).

    Returns an uninstall function.

    Parameters
    ----------
    method_families : str | Iterable[str]
        Method family names to intercept (case-insensitive).
    factory : Callable[[], ASECalculatorBridge] | None
        Zero-argument factory; defaults to ``make_bridge_from_env``.
    """
    if isinstance(method_families, str):
        method_families = (method_families,)
    families = tuple(str(m).lower() for m in method_families)
    factory = factory or make_bridge_from_env

    if _state["installed"] and not force:
        _log("already installed; skipping")
        return uninstall

    orig = {
        "load": su.core.load_system_into_calculator,
        "available": su.core.get_available_settings,
        "has": su.core.has_calculator,
        "get": su.core.get_calculator,
    }

    def load_system_into_calculator(first, *args, **kwargs):
        # Note: SettingsManager passes method_family both positionally and as a
        # keyword -- accept it leniently and de-duplicate.
        method_family = args[0] if args else kwargs.pop("method_family", "Any")
        if args:
            kwargs.pop("method_family", None)
        if str(method_family).lower() in families:
            _log(f"load_system_into_calculator: matched {method_family} "
                 f"({type(first).__name__})")
            calc = factory()
            _apply_settings(calc, kwargs)
            calc.structure = _read_structure(first)
            return calc
        return orig["load"](first, method_family, **kwargs)

    def get_available_settings(method_family, program="Any"):
        if str(method_family).lower() in families:
            _log(f"get_available_settings: matched {method_family}")
            return nnp_settings(program)
        return orig["available"](method_family, program)

    def has_calculator(method_family, program="Any"):
        if str(method_family).lower() in families:
            return True
        return orig["has"](method_family, program)

    def get_calculator(method_family, program="Any"):
        if str(method_family).lower() in families:
            _log(f"get_calculator: matched {method_family}")
            return factory()
        return orig["get"](method_family, program)

    su.core.load_system_into_calculator = load_system_into_calculator
    su.core.get_available_settings = get_available_settings
    su.core.has_calculator = has_calculator
    su.core.get_calculator = get_calculator

    _state.update(installed=True, families=families, factory=factory, originals=orig)
    _log(f"installed (method families = {families})")
    return uninstall


def uninstall() -> None:
    """Restore the original functions."""
    if not _state["installed"]:
        return
    su.core.load_system_into_calculator = _state["originals"]["load"]
    su.core.get_available_settings = _state["originals"]["available"]
    su.core.has_calculator = _state["originals"]["has"]
    su.core.get_calculator = _state["originals"]["get"]
    _state.update(installed=False, originals={})
    _log("uninstalled")


def install_from_env() -> bool:
    """Entry point for ``sitecustomize.py``: install when ``NNP_AFIR_PUFFIN``
    is truthy."""
    if os.environ.get("NNP_AFIR_PUFFIN", "") in ("", "0"):
        return False
    families = os.environ.get("NNP_AFIR_METHOD_FAMILY", DEFAULT_METHOD_FAMILY)
    install(tuple(f.strip() for f in families.split(",") if f.strip()))
    return True
