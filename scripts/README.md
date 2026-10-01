# Utility scripts

Stand-alone tools built on the `jnuscout` package.  Each script embeds a
small `sys.path` bootstrap for the `src` layout, so they can be run directly
from a source checkout:

```bash
python scripts/<script>.py --help
```

Scripts that talk to the SCINE database additionally need `pymongo` and the
`[scine]` extra; scripts that label or validate with ORCA need an ORCA
executable (`ORCA_BIN`).

## Uncertainty calibration and active learning

- **`calibrate_uq.py`** -- builds a reference set (small CHNO molecules plus
  random displacements), computes committee predictions and ORCA L2
  reference energies/forces, and calibrates the fallback threshold with
  `calibrate_threshold` (plus rank correlation, expected calibration error
  and a coverage scan).
- **`build_al_trainset.py`** -- pulls elementary steps from a SCINE
  database, samples configurations along each MEP (`spline.evaluate(t)`),
  labels them with ORCA L2, and writes the training/validation frames for
  MACE fine-tuning.
- **`resplit_trainset.py`** -- merges and re-splits an existing frame set
  stratified by composition (guards against the "training set is all C4H6,
  validation set is all C6H10" failure mode).  Standard library only.
- **`validate_finetune.py`** -- evaluates a fine-tuned model against the
  foundation model and the ORCA reference on held-out reactions, and prints
  the barrier MAE and bias decomposition.  `--baseline` (the fine-tuned
  model) is required.
- **`analyze_committee.py`** -- compares committee member combinations on a
  barrier JSON produced by the evaluation scripts; reports single-member
  MAE/bias/slope and combination statistics.

## Data and database hygiene

- **`db_snapshot.py`** -- prints, per database, the structure count,
  elementary-step count, size histogram and reaction-type histogram (the
  snapshot used for progress numbers).
- **`check_structure_identity.py`** -- independent-source cross-check of
  database structures against RDKit-perceived references (connectivity,
  radicals, bond lengths); supports label filters (`--label minima` default)
  and reports trusted counts together with exclusion reasons.  Runs RDKit in
  a separate interpreter: set `JNUSCOUT_RDKIT_PY` to a Python with RDKit
  installed; the perception helper defaults to `_rdkit_perceive.py` in this
  directory (`JNUSCOUT_RDKIT_HELPER` to override).
- **`_rdkit_perceive.py`** -- internal helper (not run directly): perceives
  connectivity and bond orders from 3D coordinates with RDKit's
  `rdDetermineBonds` and regenerates a 3D reference from the resulting
  SMILES; called by `check_structure_identity.py`.
- **`eval_barrier_spline.py`** -- barrier evaluation by the spline-endpoint
  method (`reactant = spline.evaluate(0)`, `TS = spline.evaluate(ts_position)`),
  correct for bimolecular elementary steps where paths cannot be
  concatenated from fragments; evaluates several models on the identical
  endpoints with ORCA caching.
- **`setup_h2co_nnp_db.py`** -- builds a Chemoton database with
  `method_family='nnp'` following the H2CO assembly (same gears, same AFIR
  job) and submits one exploration cycle; note the explicit Angstrom-to-Bohr
  conversion before insertion.
- **`verify_bohr_units.py`** -- checks that geometries read from the search
  database are converted from Bohr to Angstrom correctly, using a
  known-answer self-check (an n=6 reactant's nearest-neighbour distance must
  come out as a normal C-H bond length, not a broken-bond distance).

## Verification

- **`verify.sh`** -- the unified project check: `python -m compileall`
  syntax gate, then `pytest`.
