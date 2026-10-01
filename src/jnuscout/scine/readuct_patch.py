"""Let readuct use a Python calculator (NNP) by working around a return-value
conversion failure in the SCINE bindings.

## The nature of the blocker

readuct can accept a Python calculator and use it to produce results that are
exact to the last digit. The only failing step is converting
``dict[str, shared_ptr<Calculator>]`` back to Python in ``readuct.run_*_task``
(every task function returns ``tuple[dict[str, Calculator], bool]``).

Evidence:
  * Instrumented call sequences are identical in both cases: both reach
    ``_calculate_impl``. If the calculation raises, the exception propagates;
    if the calculation completes, the TypeError occurs only afterwards.
  * Reading the energy back from the readuct clone matches the ASE reference
    to 0.00e+00.

## What this module does

1. ``install_readuct_patch()`` wraps *all* ``readuct.run_*`` functions and,
   after catching that specific TypeError, rebuilds the return dictionary
   from the captured clone.
2. ``install_puffin_patch(nnp_factory)`` wraps
   ``ScineHelper.prepare_readuct_task`` from ``scine_puffin``: when the
   model's ``method_family`` matches the sentinel value, it returns our NNP
   bridge instead of the built-in SCINE calculator.

## Nature and limitations

This is a workaround, not a fix -- the SCINE binding defect remains. The patch
only applies when "the calculation has completed and only the return-value
conversion failed", therefore:
  * it only matches the specific ``TypeError`` "incompatible function arguments";
  * it only applies to task functions that return a calculator dictionary.
If a task failure is *not* in the return-value conversion, the patch would
swallow the real error as well -- when troubleshooting, first enable
``NNP_PATCH_VERBOSE=1`` for the notices, or temporarily do not install the
patch (this environment-variable name is kept for compatibility with existing
deployments).
"""
import functools
import os

import numpy as np
import scine_readuct as readuct

_VERBOSE = os.environ.get("NNP_PATCH_VERBOSE", "") not in ("", "0")

# readuct clones during argument handling and the computation actually runs on
# the clone -- the clone must be captured to retrieve the results.
_CAPTURED = []


def _capturing_clone(bridge_cls, orig_clone):
    def clone(self):
        c = orig_clone(self)
        _CAPTURED.append(c)
        return c
    return clone


def install_readuct_patch(bridge_cls, verbose=None):
    """Wrap all ``readuct.run_*`` functions. Returns an uninstall function.

    Parameters
    ----------
    bridge_cls : type
        The ``ASECalculatorBridge`` class (its ``_clone_impl`` is taken over).
    """
    verbose = _VERBOSE if verbose is None else verbose
    orig_clone = bridge_cls._clone_impl
    bridge_cls._clone_impl = _capturing_clone(bridge_cls, orig_clone)

    originals = {}

    def wrap(name, fn):
        @functools.wraps(fn)
        def patched(systems, *args, **kwargs):
            _CAPTURED.clear()
            try:
                return fn(systems, *args, **kwargs)
            except TypeError as e:
                if "incompatible function arguments" not in str(e):
                    raise
                if not _CAPTURED:
                    raise    # no clone captured -> not the failure mode we know; do not swallow it
                if verbose:
                    print(f"  [nnp-patch] {name}: bypassing the return-value "
                          f"conversion ({len(_CAPTURED)} clone(s))")
                names = args[0] if args and isinstance(args[0], (list, tuple)) else list(systems)
                return {n: _CAPTURED[0] for n in names}, True
        return patched

    for name in dir(readuct):
        if not name.startswith("run_"):
            continue
        fn = getattr(readuct, name)
        if not callable(fn):
            continue
        originals[name] = fn
        setattr(readuct, name, wrap(name, fn))

    if verbose:
        print(f"  [nnp-patch] wrapped {len(originals)} readuct task functions")

    def uninstall():
        bridge_cls._clone_impl = orig_clone
        for n, fn in originals.items():
            setattr(readuct, n, fn)

    return uninstall


def install_puffin_patch(bridge_cls, calculator_factory, sentinel="nnp"):
    """Make Puffin use the NNP bridge when ``method_family == sentinel``.

    Parameters
    ----------
    calculator_factory : Callable[[list[str]], ase.calculators.calculator.Calculator]
        Returns an ASE calculator for a list of element symbols.
    sentinel : str
        The ``model.method_family`` value that triggers the substitution.

    Returns
    -------
    uninstall : Callable[[], None]
    """
    from scine_puffin.utilities.scine_helper import ScineHelper

    orig = ScineHelper.prepare_readuct_task

    @functools.wraps(orig)
    def patched(self, structure, calculation, settings, resources, model=None):
        m = model or calculation.get_model()
        if getattr(m, "method_family", None) != sentinel:
            return orig(self, structure, calculation, settings, resources, model)

        # Run the original flow first (it writes system.xyz and handles the
        # settings), then swap the calculator for the bridge.
        systems, keys = orig(self, structure, calculation, settings, resources, model)
        atoms = structure.get_atoms()
        els = [str(e) for e in atoms.elements]
        bridge = bridge_cls(calculator_factory(els), name=sentinel)
        # Note: the structure must be set explicitly -- readuct will not do it for you.
        bridge._structure = atoms
        if _VERBOSE:
            print(f"  [nnp-patch] prepare_readuct_task -> NNP bridge ({len(els)} atoms)")
        return {k: bridge for k in keys}, keys

    ScineHelper.prepare_readuct_task = patched

    def uninstall():
        ScineHelper.prepare_readuct_task = orig

    return uninstall
