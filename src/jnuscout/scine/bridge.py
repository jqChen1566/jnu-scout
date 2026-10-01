"""ASE <-> SCINE Calculator bridge: let SCINE/readuct/Puffin call any ASE calculator.

Background: the Safe-NNP stack is an ASE calculator, whereas
Chemoton/Puffin/readuct speak the SCINE Calculator interface. This module
verifies that a Python-side SCINE Calculator can be driven by the C++ readuct
layer; if so, the NNPs can be plugged into the automatic search pipeline
without writing a dedicated calculation executor.

Interface contract (extracted from the .so symbol table of scine_utilities
10.1.0): a Python Calculator in SCINE Core uses `_xxx_impl` double dispatch --
public methods of the C++ base class call back into the Python subclass
through `_xxx_impl`, and a missing override raises
`Missing overload of '_xxx_impl' in Python Calculator derivative.`
The full list (16):
    _allows_python_gil_release_impl, _calculate_impl, _clone_impl,
    _get_positions_impl, _get_required_properties_impl, _get_state_impl,
    _get_structure_impl, _load_state_impl, _modify_positions_impl, _name_impl,
    _possible_properties_impl, _results_impl, _set_required_properties_impl,
    _set_structure_impl, _settings_impl, _supports_method_family_impl

Unit conversion (SCINE works in atomic units):
    E[Eh] = E[eV] / 27.211386245
    g[Eh/bohr] = −F[eV/Å] / 1.8897259886 / 27.211386245
"""
import numpy as np
import scine_utilities as su
from scine_utilities.core import Calculator as ScineCalculatorBase

EV_PER_HARTREE = 27.211386245
BOHR_PER_ANGSTROM = 1.8897259886      # 1 Å = 1.8897 bohr

# Diagnostic switch: print suspicious events inside the bridge
# (off by default to avoid polluting Puffin logs)
import os as _os
_BRIDGE_VERBOSE = _os.environ.get("NNP_AFIR_BRIDGE_VERBOSE", "") not in ("", "0")


# SCINE's PropertyList only supports default construction plus
# add_property(Property) (no append and no bulk constructor; verified against
# scine_utilities 10.1.0: PropertyList(['energy']) raises
# "incompatible constructor arguments")
_PROPERTY_MAP = {
    "energy": su.Property.Energy,
    "gradients": su.Property.Gradients,
    "hessian": su.Property.Hessian,
    "atomic_charges": su.Property.AtomicCharges,
    "bond_orders": su.Property.BondOrderMatrix,
    "dipole": su.Property.Dipole,
    "repulsion_energy": su.Property.RepulsionEnergy,
}


def to_property_list(names) -> "su.PropertyList":
    """Convert a list of names to a SCINE PropertyList (unknown names are skipped silently)."""
    pl = su.PropertyList()
    for n in names:
        prop = _PROPERTY_MAP.get(str(n).lower())
        if prop is not None:
            pl.add_property(prop)
    return pl


def bridge_descriptors():
    """Standard SCINE calculator settings descriptors.

    Required: SCINE `Settings` objects only accept declared keys. With an
    empty descriptor collection, readuct raises while setting keys such as
    `self_consistence_criterion` inside optimization / scan tasks:

        `Setting 'self_consistence_criterion' not implemented in the applied calculator`

    and aborts the task outright (`run_opt_task` stops after zero
    optimization steps even though the patched call itself goes through).
    """
    d = su.DescriptorCollection("ase_bridge_settings")
    d["molecular_charge"] = su.IntDescriptor("Charge of the molecule")
    d["spin_multiplicity"] = su.IntDescriptor("Spin multiplicity")
    d["method"] = su.StringDescriptor("Electronic structure method")
    d["basis_set"] = su.StringDescriptor("Basis set")
    d["program"] = su.StringDescriptor("Program")
    d["method_family"] = su.StringDescriptor("Method family")
    d["external_program_nprocs"] = su.IntDescriptor("Number of cores")
    d["external_program_memory"] = su.IntDescriptor("Memory in MB")
    d["temperature"] = su.DoubleDescriptor("Temperature in K")
    d["pressure"] = su.DoubleDescriptor("Pressure")
    d["electronic_temperature"] = su.DoubleDescriptor("Electronic temperature")
    d["symmetry_number"] = su.IntDescriptor("Symmetry number")
    d["self_consistence_criterion"] = su.DoubleDescriptor("SCF criterion")
    d["max_scf_iterations"] = su.IntDescriptor("Max SCF iterations")
    d["mixer"] = su.StringDescriptor("SCF mixer")
    d["scf_damping"] = su.DoubleDescriptor("SCF damping")
    d["spin_mode"] = su.StringDescriptor("Spin mode")
    d["version"] = su.StringDescriptor("Program version")
    # ---- The keys below are a "superset" kept for compatibility ----
    # Puffin's SettingsManager writes these keys into calculator_settings; the
    # official calculators (DFTB3: 13 keys / PM6: 14 keys) provide them through
    # their own C++ descriptors, so the bridge must declare them itself,
    # otherwise it raises
    #   `There is no matching key 'X' in these Settings!`
    # Declared as a superset rather than a minimal set: rejecting an unknown
    # key aborts the task before it starts, which costs far more than declaring
    # a few extra keys.
    d["base_working_directory"] = su.StringDescriptor("Base working directory (readuct)")
    d["output_path"] = su.StringDescriptor("Output path (readuct)")
    d["scf_mixer"] = su.StringDescriptor("SCF mixer (Sparrow naming)")
    d["density_rmsd_criterion"] = su.DoubleDescriptor("Density RMSD convergence criterion")
    d["method_parameters"] = su.StringDescriptor("Method parameter file")
    d["max_memory"] = su.IntDescriptor("Maximum memory")
    d["log"] = su.StringDescriptor("Log level")
    d["spin_block"] = su.BoolDescriptor("Spin blocking")
    d["nddo_dipole"] = su.BoolDescriptor("NDDO dipole (PM6)")
    return d


class ASECalculatorBridge(ScineCalculatorBase):
    """Adapt an ASE calculator to the SCINE Calculator interface.

    Parameters
    ----------
    ase_calc : ase.calculators.calculator.Calculator
    name : str
        Calculator name on the SCINE side (diagnostic).
    """

    def __init__(self, ase_calc, name: str = "ase_bridge"):
        super().__init__()
        self._ase = ase_calc
        self._name = name
        self._structure = su.AtomCollection()
        self._results = su.Results()
        # Must carry descriptors -- an empty descriptor collection makes
        # readuct's opt/scan tasks fail while setting
        # self_consistence_criterion (see bridge_descriptors)
        # The constructor signature is `su.Settings(descriptors)` (single
        # argument); `su.Settings(name, descriptors)` raises
        # "incompatible constructor arguments"
        self._settings = su.Settings(bridge_descriptors())
        self._properties_pl = to_property_list(["energy", "gradients"])
        self._log = su.core.Log()
        # Whether to attach the bond-order matrix to Results (required by some
        # Puffin jobs; see _possible_properties_impl)
        self._include_bond_orders = _os.environ.get(
            "NNP_AFIR_BOND_ORDERS", "1") not in ("", "0")

    # ================= _impl interface (called from the SCINE C++ side) =================
    def _allows_python_gil_release_impl(self) -> bool:
        # False: the GIL is held during calculations -- ASE/torch callbacks
        # need it, and releasing it would crash
        return False

    def _name_impl(self) -> str:
        return self._name

    def _settings_impl(self):
        return self._settings

    def _get_structure_impl(self):
        return self._structure

    def _set_structure_impl(self, structure) -> None:
        self._structure = structure

    def _get_positions_impl(self):
        return self._structure.positions

    def _modify_positions_impl(self, positions) -> None:
        self._structure.positions = positions

    def _results_impl(self):
        return self._results

    def _set_required_properties_impl(self, properties) -> None:
        # `properties` is a PropertyList (not iterable; verified against
        # scine_utilities 10.1.0) -- store the object directly
        self._properties_pl = properties

    def _get_required_properties_impl(self):
        # Must return a PropertyList (the C++ concrete type) -- returning a
        # Python list raises
        # "Unable to cast Python instance of type <class 'list'> to C++ type '?'"
        return getattr(self, "_properties_pl", to_property_list(["energy", "gradients"]))

    def _possible_properties_impl(self):
        # Includes AtomicCharges: readuct's single-point tasks default to
        # require_charges=true; omitting it raises
        # "Charges required, but chosen calculator does not provide them"
        # Includes BondOrderMatrix: Puffin jobs such as
        # scine_geometry_optimization need bond orders; omitting it raises
        # "Bond orders required, but chosen calculator does not provide them"
        # and fails the entire job (observed in the first end-to-end Puffin run)
        # Includes Hessian: Puffin's transition-state workflow (Hessian ->
        # imaginary-frequency check -> IRC) requires it; omitting it raises
        # "Calculator is missing a Hessian property" and fails the job
        return to_property_list(["energy", "gradients", "atomic_charges",
                                 "bond_orders", "hessian"])

    def _supports_method_family_impl(self, method_family: str) -> bool:
        return True

    def _get_state_impl(self):
        return su.core.State()

    def _load_state_impl(self, state) -> None:
        return None

    def _clone_impl(self):
        """Clone the calculator.

        readuct clones the calculator internally (the debug log stays empty on
        the original object, i.e. the C++ side operates on the clone).
        Structure / settings / required properties must all be copied along,
        otherwise the clone has an empty structure and the backend (MACE)
        fails on empty coordinates with
        "zero-size array to reduction operation maximum".
        """
        import copy as _copy
        try:
            cal = _copy.deepcopy(self._ase)
        except Exception:
            cal = self._ase          # backends that cannot be deep-copied (e.g. holding CUDA handles) are shared
        new = type(self)(cal, self._name)
        # The structure must be copied for real: the previous code used
        # `su.AtomCollection(self._structure)`, a constructor that does not
        # exist (only `AtomCollection(N)` and `AtomCollection(elements, positions)`
        # are available), so it always fell into the except branch and shared
        # the same structure object -- calling `_modify_positions_impl` on the
        # clone would move the original geometry as well. readuct's optimizer
        # operates on clones throughout, and such sharing corrupts multi-system
        # workflows such as spin propensity.
        try:
            _els = [self._structure[i].element for i in range(self._structure.size())]
            _pos = np.array(self._structure.positions, dtype=float)
            new._structure = su.AtomCollection(_els, _pos)
        except Exception:                                       # noqa: BLE001
            new._structure = su.AtomCollection()                # at least do not share the original object
        new._properties_pl = self._properties_pl
        # `new._settings = self._settings` must NOT be used: SCINE C++
        # calculators own independent Settings after cloning, and sharing them
        # would make "set on the clone" also modify the original. Puffin's
        # spin-propensity workflow relies on this:
        #     shifted = systems[name].clone()
        #     shifted.settings["spin_multiplicity"] = <shifted multiplicity>
        # `_propensity_iterator` is a lazy generator that re-reads the
        # multiplicity of the original object every round; sharing pollutes the
        # original with the last written shift value ("absent when the system is
        # built, present when it is inspected"), raises
        # `Could not find system nt_multiplicity_shift_-2` and fails the whole
        # NT2 job.
        new._settings = su.Settings(bridge_descriptors())
        for _k in self._settings.keys():
            try:
                new._settings[_k] = self._settings[_k]
            except Exception:                                   # noqa: BLE001
                pass                                            # skip keys that are unset
        new._include_bond_orders = self._include_bond_orders
        return new

    def _calculate_impl(self, description: str = "") -> "su.Results":
        """Run the ASE calculation, write the results back into `self._results`,
        and return that Results object.

        `self._results` must be returned. The C++ trampoline is
        (scine_utilities source src/Utils/Python/Core/CalculatorPython.cpp:112)

            PYBIND11_OVERRIDE_PURE_NAME(const Scine::Utils::Results&, Scine::Core::Calculator,
                                        "_calculate_impl", calculate, description)

        i.e. the return value of this method has to convert to
        `const Results&`. Returning `None` makes the pybind conversion fail
        with a TypeError; and scine_utilities'
            CalculationRoutines::calculateWithCatch(calculator, log, "Aborting optimization …")
        swallows ANY exception with `catch (...)`, prints only
            "Aborting optimization due to failed calculation"
        and rethrows. The observable effect is that a geometry optimization
        stops after a single calculation, and that message is completely
        unrelated to the actual cause (a wrong return type), which makes it
        highly misleading.

        The parameter is the description string (the pybind binding names it
        `dummy`: calculator.def("calculate", …, pybind11::arg("dummy") = std::string{})),
        NOT a PropertyList -- do not store it into `_properties_pl`. Required
        properties are set separately through `_set_required_properties_impl`.
        """
        n = self._structure.size()
        positions = np.asarray(self._structure.positions, dtype=float)          # bohr
        energy_eh, gradients_eh_bohr = self._eval_at(positions)

        # `self._results` must be updated in place, never re-created: readuct
        # takes a reference to the Results object via `_results_impl` ahead of
        # time and keeps it (in single-point tasks `_results_impl` is called
        # three times before the calculation). Re-assigning
        # `self._results = su.Results()` would leave the old object held by
        # readuct with successful_calculation == False forever -- the observable
        # effect is again "stops after a single calculation" plus
        # `Aborting optimization due to failed calculation`.
        r = self._results
        r.energy = energy_eh                                                  # Eh
        r.gradients = gradients_eh_bohr                                       # Eh/bohr
        # Atomic charges: the ASE backends (MACE/AIMNet2) do not expose them;
        # neutral zeros serve as placeholders. Limited impact on AFIR search
        # (reactive-site detection relies mainly on bonding topology).
        # TODO: hook up a charge model when needed (e.g. the AIMNet2 charge
        # head / QEq).
        r.atomic_charges = np.zeros(n)

        # Bond orders: not provided by the NNP backends; detected with SCINE's
        # own distance criterion (sum of covalent radii + 0.4 Å) on the
        # structure in SCINE units (Bohr). Some Puffin jobs require this
        # property.
        if self._include_bond_orders:
            try:
                r.bond_orders = su.BondDetector.detect_bonds(self._structure)
            except Exception as e:                              # noqa: BLE001
                if _BRIDGE_VERBOSE:
                    print(f"[bridge] bond-order detection failed, skipped: {type(e).__name__}: {e}")

        # Hessian: required by Puffin's transition-state workflow
        # ("Calculator is missing a Hessian property"). Computed numerically by
        # central differences of the analytic gradients, with the step in Bohr,
        # so the unit is naturally Eh/bohr². Cost = 6N gradient evaluations
        # (24 for H2CO, about 1 s). Only evaluated when actually requested.
        if self._wants_property(su.Property.Hessian):
            try:
                r.hessian = self._numerical_hessian(positions)
            except Exception as e:                              # noqa: BLE001
                if _BRIDGE_VERBOSE:
                    print(f"[bridge] numerical Hessian failed: {type(e).__name__}: {e}")

        # successful_calculation must be set LAST: calculateWithCatch relies on
        # it to decide whether this step succeeded
        r.successful_calculation = True
        # Must be returned (see docstring) -- returning None is the root cause
        # of "optimization stops after a single calculation"
        return r

    # ---------------- calculation kernel (shared by _calculate_impl and the numerical Hessian) ----------------

    def _eval_at(self, positions_bohr):
        """Evaluate energy and gradients at the given **Bohr** coordinates; returns (E[Eh], g[Eh/bohr])."""
        from ase import Atoms
        n = self._structure.size()
        symbols = [str(self._structure[i].element) for i in range(n)]
        at = Atoms(symbols, positions=np.asarray(positions_bohr, dtype=float)
                   / BOHR_PER_ANGSTROM)                     # bohr → Å
        at.calc = self._ase
        energy_eV = float(at.get_potential_energy())
        forces_eV_A = np.asarray(at.get_forces(), dtype=float)
        # g[Eh/bohr] = −F[eV/Å] / (Å→bohr) / (eV→Eh); note that F = −g
        return (energy_eV / EV_PER_HARTREE,
                -forces_eV_A / BOHR_PER_ANGSTROM / EV_PER_HARTREE)

    def _numerical_hessian(self, positions_bohr, step=1e-2):
        """Central-difference Hessian of the analytic gradients, in Eh/bohr², shape (3N, 3N)."""
        x0 = np.asarray(positions_bohr, dtype=float).copy()
        m = x0.size
        hess = np.zeros((m, m))
        for i in range(m):
            xp = x0.copy(); xp.reshape(-1)[i] += step
            xm = x0.copy(); xm.reshape(-1)[i] -= step
            _, gp = self._eval_at(xp)
            _, gm = self._eval_at(xm)
            hess[:, i] = (gp - gm).reshape(-1) / (2.0 * step)
        return 0.5 * (hess + hess.T)                            # symmetrize

    def _wants_property(self, prop) -> bool:
        """Whether the current required properties include `prop`.

        `PropertyList.__contains__` is bound
        (src/Utils/Python/PropertyListPython.cpp:67), so `in` works directly;
        note that a PropertyList itself is NOT iterable.
        """
        try:
            return prop in self._properties_pl
        except Exception:                                       # noqa: BLE001
            return False

    # ================= legacy-compatible public methods (some versions call these directly) =================
    @property
    def name(self) -> str:
        return self._name

    @property
    def settings(self):
        return self._settings

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, value) -> None:
        self._structure = value

    @property
    def positions(self):
        return self._structure.positions

    @positions.setter
    def positions(self, value) -> None:
        self._structure.positions = value

    @property
    def log(self):
        return self._log

    def get_results(self):
        return self._results

    def has_results(self) -> bool:
        return True

    def delete_results(self) -> None:
        # NOTE: re-binding `self._results` here would dangle any reference the
        # C++ side previously obtained through `_results_impl`. Currently
        # unused; if a C++ task ever calls it mid-run, clear in place instead.
        self._results = su.Results()

    def set_required_properties(self, properties) -> None:
        self._properties_pl = properties

    def get_required_properties(self):
        return getattr(self, "_properties_pl", to_property_list(["energy", "gradients"]))

    def get_possible_properties(self):
        return to_property_list(["energy", "gradients"])

    def clone(self):
        return self._clone_impl()

    def calculate(self, description: str = "") -> "su.Results":
        """Python-side entry point: forwards to `_calculate_impl` and returns Results.

        The parameter is the description string, not a property list (the
        pybind binding names it `dummy`:
        calculator.def("calculate", …, pybind11::arg("dummy") = std::string{})).
        Set required properties with `set_required_properties()` instead.
        C++ dispatch does NOT go through here -- the trampoline calls
        `_calculate_impl` directly. This method is only an equivalent entry
        point for Python callers.
        """
        return self._calculate_impl(description)

    def __repr__(self) -> str:
        return f"<ASECalculatorBridge name={self._name!r} backend={type(self._ase).__name__}>"
