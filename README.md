<div align="center">

# JNUScout

**the machine-learning-potential scout for reaction path search** -- uncertainty-aware
reaction-network exploration on the open-source SCINE stack

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.11-blue)](https://www.python.org/)
[![Release](https://img.shields.io/badge/release-v0.1.0-brightgreen)](https://github.com/jqChen1566/jnu-scout/releases)
[![Status](https://img.shields.io/badge/status-early%20development-yellow)](#status)

</div>

JNUScout drives automated reaction-path search with machine-learning
potentials (MACE, AIMNet2) inside the open-source SCINE engine (Chemoton /
Puffin), and keeps itself honest about what the potentials can and cannot do.
A committee of models from different architecture families measures its own
disagreement on every geometry; any prediction beyond a calibrated
uncertainty threshold is handed back to higher-level quantum chemistry
(ORCA).  The exploration itself, the models and the reference calculations
all run on your own machine or cluster -- nothing is uploaded, and the
framework never hides an uncertain answer behind an averaged one.

## Functionality

**Calculators** (ASE protocol; usable stand-alone, without the engine):

- `XTBCalculator` -- GFN2-xTB via the xtb command line (L1 pre-screening);
- `ORCACalculator` -- r2SCAN-D4/def2-SVP (L2, geometry + frequencies) and
  wB97M-V/def2-TZVP (L3, high-level single points);
- `NNPCommittee` -- MACE-MP-0, MACE-MPA-0 and AIMNet2-b973c, with a
  disagreement measure (per-atom force spread and within-family energy
  spread);
- `SafeNNPCalculator` -- committee + calibrated-threshold fallback to L2;
- `CalculatorFactory` -- one call per level (L0/L1/L2/L3).

**SCINE integration** (run the NNP as a method family inside the engine):

- a full ASE <-> SCINE calculator bridge (all sixteen `_*_impl` interfaces,
  clone isolation, in-place results contract);
- process-level injection patches that resolve your calculator at Puffin's
  and readuct's single calculator-resolution choke point, with every
  non-declared method family falling back to the stock behaviour.

**Uncertainty quantification** (`jnuscout.uq`):

- threshold calibration from a reference set: the low quantile of the
  disagreement among dangerous configurations bounds the false-negative
  rate by construction;
- rank correlation, expected calibration error and coverage statistics for
  auditing a calibrated threshold;
- energy-baseline alignment for cross-member energy spreads.

**Active-learning utilities** (`scripts/`):

- `build_al_trainset.py` -- sample reaction-path splines and build labelled
  training sets;
- `resplit_trainset.py` -- composition-stratified, reaction-grouped splits;
- `validate_finetune.py` -- held-out barrier validation of a fine-tuned
  model;
- `calibrate_uq.py` -- run the calibration protocol on collected data;
- plus `analyze_committee.py`, `db_snapshot.py`,
  `check_structure_identity.py`, `eval_barrier_spline.py`,
  `setup_h2co_nnp_db.py` and `verify_bohr_units.py`.

## Status

v0.1 -- **early development**.  Interfaces and report formats may still
change between releases.

## Install

```bash
pip install jnuscout                      # core: calculators (ORCA/xTB), UQ maths
pip install "jnuscout[nnp]"              # + torch, mace-torch, aimnet2
pip install "jnuscout[scine]"            # + scine-utilities/chemoton/puffin/database
pip install "jnuscout[dev]"              # + pytest
```

Requirements and notes:

- Python >= 3.11; the core needs only `ase` and `numpy`.
- **External programs**: ORCA 6.x (academic license, for L2/L3) and,
  optionally, the `xtb` executable for L1.  Point `ORCA_BIN` / `XTB_BIN`
  environment variables at the binaries if they are not on `PATH`.
- **Model weights**: on offline machines, point `MACE_MP0_PATH` /
  `MACE_MPA0_PATH` at local `.model` files; without them MACE tries to
  download its weights on first use.
- The SCINE engine integration additionally expects a running MongoDB for
  Chemoton's database layer (stock SCINE behaviour).

## Use

### First five minutes

1. **Calibrate an uncertainty threshold** (pure numpy -- runs without any
   external program).  Collect, on a reference set, the committee
   disagreement `uq` and the true error `err` of each configuration, then:

   ```python
   import numpy as np
   from jnuscout.uq.calibration import calibrate_threshold, coverage_stats, spearman_rho

   uq  = np.array([0.02, 0.05, 0.11, 0.30, 0.08, 0.45, 0.15, 0.60])
   err = np.array([0.01, 0.03, 0.09, 0.40, 0.02, 0.55, 0.12, 0.70])  # eV/A

   theta = calibrate_threshold(uq, err, epsilon_crit=0.1)
   print(theta)                             # 0.1725 with these numbers
   print(spearman_rho(uq, err))             # 0.98 (rank correlation, want > 0.6)
   print(coverage_stats(uq, err, theta, epsilon_crit=0.1))
   ```

2. **Evaluate a geometry with the safe calculator** (needs the `[nnp]`
   extra; falls back to ORCA only if a fallback is triggered):

   ```python
   from ase.io import read
   from jnuscout.calculators import SafeNNPCalculator

   atoms = read("molecule.xyz")
   calc = SafeNNPCalculator(theta_force=0.30)        # theta from step 1
   atoms.calc = calc
   print(atoms.get_potential_energy(), atoms.get_forces())
   print(calc.n_fallback, "/", calc.n_calls)
   ```

3. **Run the engine with the NNP as a method family**: install the
   injection patch in your Puffin processes (a `sitecustomize.py` on
   `PYTHONPATH` calling `install_from_env()`, with `NNP_AFIR_PUFFIN=1`) and
   start your Chemoton exploration as usual.  The design and the exact
   choke points are described in `docs/architecture.md`.

### What the workflow looks like

```
        xTB (L1)            NNP committee (L0)                ORCA (L2/L3)
  pre-screen geometry  ->  explore paths fast  ->  fallback when uncertain
                                                          |
                                              reference barriers, training
                                              labels, final validation
```

A typical active-learning round: sample the explored paths, label frames
with ORCA L2, fine-tune the potential, and validate on held-out reactions;
`docs/method_chain.md` documents the protocol and the pitfalls (training
composition determines generality; reaction-type coverage is a separate
axis; report errors by system size, not only in aggregate).

## Documentation

- `docs/architecture.md` -- the three-layer design, the SCINE bridge
  contract and the injection patches;
- `docs/method_chain.md` -- method chain, uncertainty definitions,
  calibration protocol and the active-learning loop.

## Scope and guarantees

- the framework does not run your exploration for you: it supplies the
  calculators and the integration, and you submit jobs to your own engine;
- your files are read-only: products are new files next to the inputs;
- everything runs locally -- no network calls, no uploaded structures;
  model weights are read from local files when configured;
- the uncertainty score is a measured quantity: it is calibrated against
  reference errors and its coverage is reported, not assumed.

## Author

JNUScout is developed by Jianqi Chen and Juan Li, College of Chemistry and
Materials Science, Jinan University, Guangzhou, China.

## Funding

This project is supported by the Jinan University Provincial College
Students' Innovation and Entrepreneurship Training Program
(Project No. S202610559049).

## Citing

If you use this software, cite it as described in `CITATION.cff`:

```bibtex
@software{jnuscout,
  author  = {Jianqi Chen and Juan Li},
  title   = {{JNUScout}: the machine-learning-potential scout for reaction path search},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/jqChen1566/jnu-scout},
  license = {Apache-2.0},
}
```

## License

Apache-2.0 (see `LICENSE` and `NOTICE`).

## Development

```bash
# unified check: 3.11 syntax gate, then the test suite
bash scripts/verify.sh
```
