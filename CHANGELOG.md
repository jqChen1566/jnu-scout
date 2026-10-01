# Changelog

All notable changes to this project are documented in this file.
The format follows Keep a Changelog, and this project adheres to
semantic versioning.

## [0.1.0] - 2026-10-01

Initial release.

### Added

- ASE calculators: ORCA (r2SCAN-D4 / def2-SVP at L2, wB97M-V / def2-TZVP at
  L3), xTB (GFN2-xTB at L1), an NNP committee (MACE-MP-0, MACE-MPA-0,
  AIMNet2-b973c) with a disagreement-based uncertainty measure, and a safe
  fallback calculator that routes uncertain predictions to L2.
- SCINE integration: a full ASE <-> SCINE calculator bridge (all sixteen
  `_*_impl` interfaces) and process-level injection patches that let Puffin
  and readuct resolve the NNP as a method family.
- Uncertainty quantification: threshold calibration from a reference set,
  coverage scanning, and out-of-distribution checks.
- Utility scripts: active-learning training-set construction from search
  paths, stratified re-splitting, fine-tune validation, database snapshots,
  structure identity checks, spline-endpoint barrier evaluation, and
  committee analysis.
- Example configurations for H2CO isomerization and a bimolecular
  Diels-Alder exploration.
- A complete user manual (XeLaTeX source under `docs/manual/`, compiled
  PDF attached to the release).
