# Architecture

JNUScout is organised as three layers around one idea: run the expensive
exploration with machine-learning potentials, but only trust them where they
can be shown to be trustworthy.

```
            +-------------------------------------------------------+
            |  credibility layer                                    |
            |  committee disagreement (UQ)  ·  safe fallback to L2  |
            +-------------------------------------------------------+
                                   ^
            +-------------------------------------------------------+
            |  engine layer                                         |
            |  reaction-network exploration (SCINE Chemoton)        |
            |  job execution daemon (SCINE Puffin)                  |
            |  NNP registered as a method family  <- this package   |
            +-------------------------------------------------------+
                                   ^
            +-------------------------------------------------------+
            |  backend layer                                        |
            |  ML potentials (MACE, AIMNet2)  ·  xTB  ·  ORCA       |
            +-------------------------------------------------------+
```

## Backend layer

`jnuscout.calculators` provides ASE calculators for every method in the
chain.  All of them follow the ASE `Calculator` protocol, so any of them can
be used stand-alone (geometry optimisation with ASE, single-point scans,
custom workflows) without the SCINE engine.

| Class | Level | Method | Role |
|---|---|---|---|
| `XTBCalculator` | L1 | GFN2-xTB (xtb CLI) | fast pre-screening, initial geometries |
| `ORCACalculator` | L2 | r2SCAN-D4/def2-SVP (RI-approximated, `TightSCF`) | geometry optimisation and frequencies |
| `ORCACalculator` | L3 | wB97M-V/def2-TZVP (RIJCOSX, `defgrid3`) | high-level single points |
| `NNPCommittee` | -- | MACE-MP-0, MACE-MPA-0, AIMNet2-b973c | fast potential with a disagreement measure |
| `SafeNNPCalculator` | L0 | committee + automatic fallback | potential you can trust-conditionally |

`CalculatorFactory.create(level)` returns the calculator for a level; L0 is
the safe fallback calculator, whose threshold has to come from calibration
(see `docs/method_chain.md`) rather than a hard-coded default.

Notes on the ML potentials:

- committee members are deliberately from **different architecture
  families**; the disagreement between families is the uncertainty signal.
  Members from the same family (for example three differently seeded
  fine-tunes) produce near-identical predictions and cannot serve as an
  uncertainty measure.
- the AIMNet2 base is the `b973c` variant.  The `wb97m_0` variant shows a
  force-directionality defect on test molecules (several molecules had
  anti-parallel or orthogonal force vectors against ORCA; `b973c` was
  directionally correct on the same set), so `b973c` enters the committee.
- model weights are looked up through environment variables
  (`MACE_MP0_PATH`, ...) so that offline machines can point at local files
  instead of downloading.

## Engine layer: driving SCINE with a Python potential

SCINE's exploration engine (Chemoton) and its job daemon (Puffin) never
receive calculator objects.  They resolve calculators from a **method-family
string** at a single choke point:

```
scine_puffin/utilities/scine_helper.py
    SettingsManager.prepare_readuct_task(...)
        system = utilities.core.load_system_into_calculator(
            "system.xyz", model.method_family, **calculator_settings)
```

To make the engine run NNP/ORCA/xTB evaluations, this package intercepts
that resolution step and returns Python calculators implementing SCINE's
`Calculator` interface.  Two building blocks:

### The bridge (`jnuscout.scine.bridge`)

`ASECalculatorBridge` implements the full SCINE Python-calculator contract:
sixteen `_*_impl` methods, resolved through the double dispatch that the C++
base class performs on Python subclasses.  Getting this contract right is
the whole game; three points are worth calling out because the failure modes
are silent:

- **Return types are part of the contract.**  `_calculate_impl` must return
  the `Results` object; returning `None` makes the pybind11 layer raise, the
  engine's `catch (...)` swallow the error, and the optimisation stop after
  a single evaluation with a generic "failed calculation" message that
  points nowhere near the cause.
- **Update `self._results` in place.**  The C++ side captures a reference to
  the results object before the calculation; replacing the attribute with a
  fresh object leaves that reference pointing at stale data.
- **Clones must be fully isolated.**  Puffin derives spin-shifted systems by
  cloning a calculator and writing new settings into the clone; sharing the
  settings object would silently modify the original.  The bridge copies
  settings and structure on `_clone_impl`.

### The injection patches (`jnuscout.scine.puffin_patch`, `readuct_patch`)

`puffin_patch.install_from_env()` replaces two functions on
`scine_utilities.core` — `load_system_into_calculator` and
`get_available_settings` — for the declared method families only; every
other method family falls back to the original function, so Sparrow/DFTB3
and friends keep working unchanged.  The patch is meant to be activated from
a `sitecustomize.py` on `PYTHONPATH`, which guarantees it is installed in
every process of the job daemon, including forked workers, **before** Puffin
starts.

The patch declares a **superset** of calculator properties and settings
descriptors (energy, gradients, atomic charges, bond orders, Hessians; the
full standard settings key set).  The engine fails the whole job when a
property it requires was not declared, so over-declaring is cheap and
under-declaring is fatal.

Environment variables (`NNP_AFIR_PUFFIN`, `NNP_AFIR_METHOD_FAMILY`, ...)
keep their historical names for compatibility with existing deployments.

## Credibility layer

The committee's disagreement is the uncertainty score; a calibrated
threshold decides, at run time and per configuration, whether the NNP answer
is used or the calculation falls back to ORCA L2.  The mechanism, the
disagreement definitions and the calibration protocol live in
`docs/method_chain.md`.

Design notes kept for the record:

- energy disagreement is computed within one model family only: absolute
  energies across families have different zeros (atomic-reference vs
  absolute), so cross-family energy differences are meaningless.
- force disagreement is a per-atom maximum, not a mean: in reacting
  geometries the failure is often a single atom with a wrong force
  direction, which averaging would hide.

## What runs where

Nothing in this package runs a job for you: it supplies the calculators and
the integration glue inside your engine processes.  Jobs are submitted to
your own SCINE Puffin daemon; QC evaluations happen in external programs
(ORCA, xtb) on your own machine or cluster.
