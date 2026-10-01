"""Build a Chemoton database with method_family='nnp' and run one AFIR exploration cycle.

Follows the DFTB3 H2CO assembly (same set of gears, same AFIR job) with two
deliberate changes:
  1. ``db.Model('dftb3', 'dftb3', '')`` -> ``db.Model('nnp', 'nnp', '')``
  2. coordinates are defined in Angstrom and explicitly converted to Bohr
     before insertion (the database stores ``structures.atoms`` in Bohr;
     inserting Angstrom values directly would make them be read as Bohr and
     compress every geometry by a factor of 1.8897)

Puffin (with the NNP patch enabled) must already be running:

    NNP_AFIR_PUFFIN=1 NNP_AFIR_BACKEND=mace NNP_AFIR_MODEL=<model> \
      puffin start -c <puffin.yaml>

(see docs/architecture.md for the sitecustomize.py injection)

Usage:
    python scripts/setup_h2co_nnp_db.py [--run] [--wipe]
"""
import argparse

import numpy as np
import scine_database as db
import scine_utilities as su

BOHR_PER_ANGSTROM = 1.8897259886
DB_NAME = "h2co_nnp"
METHOD_FAMILY = "nnp"

# H2CO: C=O 1.20 A, C-H 1.09 A, HCH 120 deg (defined in Angstrom,
# converted to Bohr before insertion)
POS_ANG = np.array([[0.000, 0.000, 0.0],
                    [1.200, 0.000, 0.0],
                    [-0.545, 0.944, 0.0],
                    [-0.545, -0.944, 0.0]])
ELEMENTS = [su.ElementType.C, su.ElementType.O, su.ElementType.H, su.ElementType.H]


def connect():
    cred = db.Credentials()
    cred.hostname, cred.port, cred.database_name = "127.0.0.1", 27017, DB_NAME
    manager = db.Manager()
    manager.set_credentials(cred)
    manager.connect()
    return cred, manager


def main():
    global DB_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="run one cycle right after assembly")
    ap.add_argument("--wipe", action="store_true", help="wipe the database first")
    ap.add_argument("--only-run", action="store_true",
                    help="skip structure insertion, only run the gears for --loops rounds "
                         "(housekeeping needs several rounds to produce compounds)")
    ap.add_argument("--loops", type=int, default=1, help="number of gear cycles (default 1)")
    ap.add_argument("--db", default=DB_NAME, help="database name (default: h2co_nnp)")
    a = ap.parse_args()

    DB_NAME = a.db

    from scine_chemoton.utilities.insert_initial_structure import insert_initial_structure
    from scine_chemoton.filters.aggregate_filters import AggregateFilter
    from scine_chemoton.filters.further_exploration_filters import FurtherExplorationFilter
    from scine_chemoton.gears.compound import BasicAggregateHousekeeping
    from scine_chemoton.gears.elementary_steps.minimal import MinimalElementarySteps
    from scine_chemoton.gears.elementary_steps.trial_generator.bond_based import BondBased
    from scine_chemoton.gears.network_refinement.calculation_based_refinement import CalculationBasedRefinement
    from scine_chemoton.gears.scheduler import Scheduler
    from scine_chemoton.engine import Engine

    cred, manager = connect()
    if a.wipe:
        try:
            manager.wipe()
            print("database wiped")
        except Exception as e:                                  # noqa: BLE001
            print(f"wipe: {e}")
    manager.init()

    # ---- 1. initial structure, model = ('nnp', 'nnp', '') ----
    model = db.Model("nnp", "nnp", "")
    if not a.only_run:
        atoms = su.AtomCollection(ELEMENTS, POS_ANG * BOHR_PER_ANGSTROM)
        insert_initial_structure(manager, atoms, 0, 1, model, label=db.Label.MINIMUM_GUESS)
        print(f"inserted H2CO with model = (method={model.method!r}, "
              f"family={model.method_family!r}, program={model.program!r})")
    else:
        print("(--only-run: skipping structure insertion)")

    # ---- 2. gears (same assembly as the DFTB3 H2CO example) ----
    # NT2 must be used here, not AFIR: the BondBased unimolecular path calls
    # _add_reactive_complex_calculation, which only accepts NT2 job orders and
    # otherwise raises
    #   RuntimeError: Only 'NT2'-based job orders are supported for
    #                 bond-based reactive complex calculations.
    afir_job = db.Job("scine_react_complex_nt2")
    afir_settings = su.ValueCollection({})

    es_gear = MinimalElementarySteps()
    es_gear.trial_generator = BondBased()
    # Both must be set: the gear's own options.model overrides the trial
    # generator's (otherwise a ModelChangedWarning is raised and the gear
    # default -- quite possibly not 'nnp' -- is used).
    es_gear.options.model = model
    es_gear.trial_generator.options.model = model
    un = es_gear.trial_generator.options.unimolecular_options
    un.min_bond_modifications = un.max_bond_modifications = 2
    un.min_bond_formations = un.max_bond_formations = 1
    un.min_bond_dissociations = un.max_bond_dissociations = 1
    un.job = afir_job
    un.job_settings_associative = afir_settings
    un.job_settings_disconnective = afir_settings
    un.job_settings_dissociative = afir_settings
    bi = es_gear.trial_generator.options.bimolecular_options
    bi.min_bond_modifications = bi.max_bond_modifications = 1
    bi.min_inter_bond_formations = bi.max_inter_bond_formations = 1
    bi.job = afir_job
    bi.job_settings = afir_settings
    es_gear.trial_generator.reactive_site_filter = FurtherExplorationFilter()
    es_gear.aggregate_filter = AggregateFilter()

    comp_gear = BasicAggregateHousekeeping()
    comp_gear.options.model = model
    ref_gear = CalculationBasedRefinement()
    ref_gear.options.model = model
    sched_gear = Scheduler()

    if not a.run:
        print("assembly complete (not run). Add --run to execute one cycle.")
        return

    # ---- 3. run ----
    comp_coll = manager.get_collection("compounds")

    def enable_all_exploration(tag: str):
        """Chemoton-created compounds default to exploration_disabled=True, so the
        elementary-steps gear cannot match them and the automatic search silently
        produces nothing.  Re-enable every cycle."""
        n = 0
        for c in comp_coll.iterate_all_compounds():
            c.link(comp_coll)
            try:
                if not c.explore():
                    c.enable_exploration()
                    n += 1
            except Exception:                                   # noqa: BLE001
                pass
        if n:
            print(f"  [{tag}] enabled exploration on {n} compounds")

    eng = Engine(cred, fork=False)
    for rnd in range(1, a.loops + 1):
        for name, gear in (("aggregate housekeeping", comp_gear),
                           ("elementary steps", es_gear),
                           ("refinement", ref_gear),
                           ("scheduler", sched_gear)):
            print(f"  [{rnd}/{a.loops}] {name} ...")
            eng.set_gear(gear)
            eng.run(single=True)
            if name == "aggregate housekeeping":
                enable_all_exploration(f"{rnd}/{a.loops}")
    print("cycle finished")

    # ---- 4. result counts ----
    print("collection counts:")
    for coll in ("compounds", "structures", "elementary_steps", "reactions",
                 "calculations", "flasks"):
        try:
            c = manager.get_collection(coll)
            print(f"  {coll:<20} {c.count('{}')}")
        except Exception as e:                                  # noqa: BLE001
            print(f"  {coll:<20} (failed to read: {type(e).__name__}: {str(e)[:60]})")


if __name__ == "__main__":
    main()
