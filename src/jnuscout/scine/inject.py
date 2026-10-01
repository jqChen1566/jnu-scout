"""Calculator factory-level patch: let Chemoton/Puffin use the jnuscout ASE calculators.

Replaces two resolution functions on scine_utilities.core (effective only for
the method families of this package; every other method_family falls back to
the original function, so modules such as sparrow are unaffected):

1. load_system_into_calculator(path, method_family, **kwargs)
   -> returns a ScineORCACalculator (L1/L2/L3 chosen from the settings)
2. get_available_settings(method_family, program)
   -> returns our settings descriptors (needed by SettingsManager)

Usage (in any process -- the chemoton main process or the puffin daemon):
    import jnuscout.scine as scine
    scine.apply_patch()
"""

from __future__ import annotations

import logging

import scine_utilities as su

from .calculator import ScineORCACalculator, _build_descriptors

logger = logging.getLogger(__name__)

# Method families declared by this package (the L1/L2/L3 method chain)
OUR_METHOD_FAMILIES = ("DFT", "GFN2-xTB")

_patched = False


def _make_settings_for(family: str, program: str) -> su.Settings:
    """Build a descriptor-carrying Settings object (including the program key, which SettingsManager sets)."""
    descr = _build_descriptors()
    settings = su.Settings(descr)
    if "program" in settings.keys():
        settings["program"] = program
    return settings


def load_system_into_calculator(path: str, *args, **kwargs) -> su.core.Calculator:
    """Replacement: our method families -> ScineORCACalculator; everything else -> original function.

    Note: SettingsManager's calculator_settings contains a ``method_family``
    key, so a call of the form
    `load_system_into_calculator(xyz, method_family, **settings)` passes the
    argument both positionally and by keyword -- *args accepts this leniently
    and the duplicate keyword is dropped.
    """
    method_family = args[0] if args else kwargs.pop("method_family", "DFT")
    kwargs.pop("method_family", None)  # de-duplicate
    if method_family in OUR_METHOD_FAMILIES:
        logger.info("jnuscout: load_system_into_calculator(%s, %s, %s)",
                    path, method_family, kwargs)
        calc = ScineORCACalculator(tier="L2")
        # Apply settings kwargs (basis_set/method determine the tier)
        for k, v in kwargs.items():
            calc.update_settings(calc.settings, k, v)
        return calc
    return _ORIGINAL_LOAD(path, method_family, **kwargs)


def get_available_settings(method_family: str, program: str = "Any") -> su.Settings:
    """Replacement: our method families -> our descriptor Settings; everything else -> original function."""
    if method_family in OUR_METHOD_FAMILIES:
        return _make_settings_for(method_family, program)
    return _ORIGINAL_GET_AVAILABLE(method_family, program)


_ORIGINAL_LOAD = None
_ORIGINAL_GET_AVAILABLE = None


def apply_patch(force: bool = False) -> None:
    """Replace the two factory functions on scine_utilities.core (idempotent)."""
    global _patched, _ORIGINAL_LOAD, _ORIGINAL_GET_AVAILABLE
    if _patched and not force:
        return
    _ORIGINAL_LOAD = su.core.load_system_into_calculator
    _ORIGINAL_GET_AVAILABLE = su.core.get_available_settings
    su.core.load_system_into_calculator = load_system_into_calculator
    su.core.get_available_settings = get_available_settings
    _patched = True
    logger.info("jnuscout scine patch applied (families=%s)", OUR_METHOD_FAMILIES)
