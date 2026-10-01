"""JNUScout -- the machine-learning-potential scout for reaction path search.

A framework that drives open-source reaction-network exploration (SCINE
Chemoton / Puffin) with machine-learning potentials (MACE, AIMNet2), and
falls back to high-level quantum chemistry whenever the models disagree
beyond a calibrated threshold.

Subpackages are imported lazily on purpose: the top-level import must stay
cheap and must not require the heavy optional stacks (torch / mace / scine).

    jnuscout.calculators -- ASE calculators: ORCA (L2/L3), xTB (L1),
                            NNP committee, safe fallback (L0), factory
    jnuscout.scine       -- ASE <-> SCINE bridge and process-level injection
                            patches that let Puffin / readuct run the NNP
    jnuscout.uq          -- uncertainty threshold calibration and coverage
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
