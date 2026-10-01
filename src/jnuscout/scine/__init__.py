"""SCINE calculator injection layer (Chemoton/Puffin <-> jnuscout ASE Calculator).

Background: all Chemoton 4.1.0 calculations are executed through an external
daemon (scine_puffin), and puffin's ReactJob resolves a calculator by
method_family through ``scine_utilities.core.load_system_into_calculator``.
The scine_utilities 10.1.0 bindings used here do not support registering a
pure Python Module (the Module subclass aborts on a pure virtual call during
construction; sparrow-style modules are C++ .module.so), so this layer uses a
**factory-level patch**: it replaces only the calculator resolution functions
(load_system_into_calculator / get_available_settings) and returns Python
subclasses implementing the official ``scine_utilities.core.Calculator``
interface. The AFIR engine (Chemoton fragment_based + Readuct) and the
database layer (scine_database/MongoDB) are left untouched.

The recommended entry point is ``puffin_patch``: the
``ScineORCACalculator`` used by ``inject.py`` implements only 5 of the 16
required ``_*_impl`` methods and lacks ``_clone_impl``, so that chain is
incomplete by itself; ``ASECalculatorBridge`` implements all 16 and has been
validated end to end (a real Puffin job reached ``Status.COMPLETE``).
``inject.py`` is kept for comparison.
"""

from .calculator import ScineORCACalculator
from .inject import (
    OUR_METHOD_FAMILIES,
    apply_patch,
    get_available_settings,
    load_system_into_calculator,
)
from .puffin_patch import install, install_from_env, make_bridge_from_env, uninstall

__all__ = [
    "ScineORCACalculator",
    "OUR_METHOD_FAMILIES",
    "apply_patch",
    "get_available_settings",
    "load_system_into_calculator",
    # recommended entry points
    "install",
    "install_from_env",
    "uninstall",
    "make_bridge_from_env",
]
