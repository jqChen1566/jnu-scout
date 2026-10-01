# Method chain, uncertainty quantification and active learning

## The method chain

| Level | Method | Auxiliary | Program | Role |
|---|---|---|---|---|
| L1 | GFN2-xTB | -- | xtb (CLI) | fast pre-screening, initial geometry guesses |
| L2 | r2SCAN-D4 / def2-SVP | def2/J | ORCA | geometry optimisation and frequencies (production level) |
| L3 | wB97M-V / def2-TZVP | def2/JK | ORCA | high-level single points |
| L0 | ML potential + fallback | -- | MACE / AIMNet2 (+ ORCA L2) | exploration with run-time trust control |

L2 and L3 are ordinary external-program calls wrapped as ASE calculators
(`jnuscout.calculators.orca`).  L0 is the interesting one: it runs the ML
potential, measures how much the models disagree, and hands the evaluation
back to L2 when the disagreement exceeds a calibrated threshold.

## Uncertainty quantification

### Disagreement definitions

The committee is a set of ML potentials from **different architecture
families** (MACE-MP-0, MACE-MPA-0, AIMNet2-b973c).  Two scores are computed
per configuration (`NNPCommittee.disagreement`):

- **Force disagreement** — the per-atom deviation of each member from the
  committee mean force, mapped to its vector norm, then maximised over all
  atoms and members:

  ```
  uq_F = max_{atom j, member i} || F_i,j - mean_i(F_i,j) ||      [eV/A]
  ```

  The maximum, not the mean, is deliberate: in reacting geometries the
  failure mode is typically one atom with a wrong force direction, and a
  mean over atoms would dilute it.

- **Energy disagreement** — the standard deviation of the members' energies,
  divided by the atom count:

  ```
  uq_E = std_i(E_i) / N                                          [eV/atom]
  ```

  Energy disagreement is only computed **within one backend family**
  (by default the largest family, the two MACE models).  Cross-family
  absolute energies are not comparable: MACE reports atomization energies
  (H2O around -14 eV) while AIMNet2 reports absolute total energies (H2O
  around -2079 eV), so subtracting across families is meaningless.

- **Energy baseline alignment** — optional per-member, per-atom baseline
  offsets (`energy_baselines`) can be supplied to remove systematic energy
  offsets between members; without them the raw absolute-energy
  disagreement is diagnostics-only.

### Threshold calibration

`jnuscout.uq.calibration.calibrate_threshold(uq, err, epsilon_crit)`
implements the protocol:

```
theta = Quantile( uq | err > epsilon_crit , 5% )
```

Take the *low quantile* of the uncertainty score **among the dangerous
configurations** (those whose true error exceeds `epsilon_crit`).  Only 5%
of dangerous configurations sit below theta, so the false-negative rate of
the fallback trigger is bounded at 5% by construction.

This is not the same as taking the 95th percentile of the uq distribution
itself: that threshold describes the high tail of the score, which may fall
between two modes and catch dangerous configurations arbitrarily.

Empirical sanity checks for a calibrated threshold are provided by
`spearman_rho` (rank correlation between uq and the true error; a usable
committee reaches rho well above 0.6) and `expected_calibration_error`;
`coverage_stats` reports, for a threshold, the fallback rate and the
fraction of dangerous configurations kept.

### Coverage is a consequence of the accuracy requirement

The fallback rate is not a fixed performance number of the method.  It
follows from the accuracy target: a stricter `epsilon_crit` (more
configurations labelled dangerous) forces a lower theta and a higher
fallback rate.  Quote the pair, never the rate alone.

## Safe fallback

`SafeNNPCalculator` runs the committee once per configuration and routes:

```
uq <  theta          -> use the committee mean           (fast path)
uq >= theta          -> evaluate with ORCA L2            (trusted path)
```

Both criteria (force and energy) are supported; by default either one
exceeding its threshold triggers the fallback (`use_or_rule=True`, the
conservative choice).  The fallback calculator is constructed lazily — no
ORCA process exists unless a fallback actually happens — and
`n_fallback / n_total` is tracked for auditing.

## Active learning loop

The training protocol that produced the fine-tuned potentials:

1. **Sample** — evaluate the path splines of explored reactions at uniform
   fractions of the reaction coordinate (`evaluate(t)`), which yields
   complete geometries including the transition-state region.
2. **Label** — run ORCA L2 (r2SCAN-D4/def2-SVP) on the sampled frames and
   keep energy and forces.
3. **Split** — stratify by composition **and group by reaction**, so that
   no reaction appears in both the training and the held-out set.
4. **Fine-tune** — MACE fine-tuning on the labelled frames; validation on
   the held-out reactions, never on random frames of trained reactions.
5. **Verify** — re-score the held-out reactions' barriers and check both the
   mean absolute error and its decomposition by composition and by reaction
   type.

Lessons that shaped the protocol:

- Training composition determines generality.  A single-composition training
  set produces a specialist: excellent inside its composition, worse than
  the untuned foundation model outside it.  Extending the training
  composition removes that specialization.
- Reaction-type coverage is separate from composition coverage.  A model
  fine-tuned on single-molecule rearrangements can still degrade on
  bimolecular reactions even after composition extension; the
  out-of-distribution check is part of the release protocol, not an optional
  extra.
- Model accuracy is not uniform across system size.  Report error by size
  bin; an aggregate mean hides systematically weak bins.

## Measured record

Results observed with this chain on the development systems (C/H/N/O
chemistry; ORCA L2 as the reference; barriers in kcal/mol):

- fine-tuned potential, held-out reactions with cross-composition training:
  barrier MAE 2.92 (against 6.93 for the strongest un-tuned foundation
  model on the same benchmark);
- committee disagreement vs true force error: rank correlation 0.847 at
  calibration; the fallback trigger fired preferentially in transition-state
  regions, where the disagreement rose by several times over equilibrium
  values.

These numbers characterise the protocol on the tested systems; they are not
transferable accuracy guarantees for other chemistries.
