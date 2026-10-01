"""ScineORCACalculator -- a Python implementation of the official SCINE Calculator interface.

Wraps the jnuscout ASE calculators (L2/L3 = ORCA; L1 = xTB) so that
Chemoton/Readuct can call them directly. Units: energy = Hartree,
gradients = Eh/bohr (SCINE convention).

Settings keys (descriptors declared in _build_descriptors):
  molecular_charge, spin_multiplicity, method, basis_set, program,
  method_family, external_program_nprocs, external_program_memory,
  temperature, pressure, electronic_temperature, symmetry_number,
  self_consistence_criterion, max_scf_iterations, mixer, scf_damping,
  spin_mode, version
"""

from __future__ import annotations

import numpy as np
import scine_utilities as su

from jnuscout.calculators import ORCACalculator, XTBCalculator

# Units
HA2EV = 27.211386245988
BOHR2ANG = 0.529177210903

# Element map (scine ElementType -> ASE symbol)
_ELEMENT_TO_SYMBOL = {
    su.ElementType.H: "H", su.ElementType.He: "He", su.ElementType.Li: "Li",
    su.ElementType.Be: "Be", su.ElementType.B: "B", su.ElementType.C: "C",
    su.ElementType.N: "N", su.ElementType.O: "O", su.ElementType.F: "F",
    su.ElementType.Ne: "Ne", su.ElementType.Na: "Na", su.ElementType.Mg: "Mg",
    su.ElementType.Al: "Al", su.ElementType.Si: "Si", su.ElementType.P: "P",
    su.ElementType.S: "S", su.ElementType.Cl: "Cl", su.ElementType.Ar: "Ar",
}


def _build_descriptors() -> su.DescriptorCollection:
    """Declare the keys and types of calculator_settings (mirrors scine settings_names)."""
    d = su.DescriptorCollection("jnuscout_calculator_settings")
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
    return d


def _tier_from_settings(settings: su.Settings, default_tier: str = "L2") -> str:
    """Map model settings (basis_set/method/method_family) to a method-chain tier."""
    method_family = str(settings.get("method_family", "")) if "method_family" in settings.keys() else ""
    basis = str(settings.get("basis_set", "")) if "basis_set" in settings.keys() else ""
    method = str(settings.get("method", "")) if "method" in settings.keys() else ""
    fam = (method_family or "").lower()
    if "xtb" in fam or "gfn" in fam:
        return "L1"
    if "tzvp" in basis.lower() or "wb97m" in method.lower() or "wB97M" in method:
        return "L3"
    return "L2"


class ScineORCACalculator(su.core.Calculator):
    """Wrap a jnuscout ASE calculator as a SCINE Calculator (energy + gradients)."""

    def __init__(self, tier: str = "L2", nprocs: int = 8, maxcore: int = 2000):
        su.core.Calculator.__init__(self)
        self._tier = tier
        self._nprocs = nprocs
        self._maxcore = maxcore
        self._charge = 0
        self._multiplicity = 1
        self._structure: su.AtomCollection | None = None
        self._descriptors = _build_descriptors()
        self._settings = su.Settings(self._descriptors)
        self._required_properties = [su.Property.Energy, su.Property.Gradients]

    # ------------------------------------------------------------------
    # Internal: structure -> ASE Atoms
    def _to_ase_atoms(self):
        from ase import Atoms

        if self._structure is None:
            raise RuntimeError("ScineORCACalculator: no structure set")
        symbols = [_ELEMENT_TO_SYMBOL[el] for el in self._structure.elements]
        pos = np.asarray(self._structure.positions) * BOHR2ANG  # bohr -> A
        atoms = Atoms(symbols, positions=pos)
        atoms.set_initial_charges([self._charge] * len(atoms))
        return atoms

    def _build_ase_calculator(self):
        if self._tier == "L1":
            return XTBCalculator(charge=self._charge, multiplicity=self._multiplicity,
                                 nprocs=min(self._nprocs, 4))
        return ORCACalculator(tier=self._tier, charge=self._charge,
                              multiplicity=self._multiplicity,
                              nprocs=self._nprocs, maxcore=self._maxcore)

    # ------------------------------------------------------------------
    # SCINE Calculator interface (impl virtuals)
    def _settings_impl(self):
        return self._settings

    def _set_structure_impl(self, structure):
        self._structure = structure

    def _get_structure_impl(self):
        return self._structure

    def _get_positions_impl(self):
        if self._structure is None:
            return np.zeros((0, 3))
        return np.asarray(self._structure.positions)

    # ------------------------------------------------------------------
    # SCINE Calculator interface (public methods)
    def calculate(self, dummy: str = "") -> su.Results:
        atoms = self._to_ase_atoms()
        calc = self._build_ase_calculator()
        atoms.calc = calc
        e_ev = atoms.get_potential_energy()
        f_ev_ang = atoms.get_forces()
        e_ha = e_ev / HA2EV
        grad_ha_bohr = -f_ev_ang / (HA2EV / BOHR2ANG)  # Eh/bohr
        results = su.Results()
        results.energy = e_ha
        results.gradients = np.asarray(grad_ha_bohr, dtype=float)
        results.successful_calculation = True
        results.program_name = "jnuscout"
        return results

    def name(self) -> str:
        return f"jnuscout_{self._tier}"

    def copy(self) -> "ScineORCACalculator":
        return ScineORCACalculator(tier=self._tier, nprocs=self._nprocs,
                                   maxcore=self._maxcore)

    def get_possible_properties(self):
        return su.core.PropertyList([su.Property.Energy, su.Property.Gradients])

    def get_required_properties(self):
        return su.core.PropertyList(self._required_properties)

    def set_required_properties(self, properties):
        self._required_properties = list(properties)

    def update_settings(self, settings: su.Settings, key: str, value):
        """Sync a settings entry to the internal ASE calculator parameters."""
        settings[key] = value
        if key == "molecular_charge":
            self._charge = int(value)
        elif key == "spin_multiplicity":
            self._multiplicity = int(value)
        elif key in ("basis_set", "method", "method_family"):
            self._tier = _tier_from_settings(settings, default_tier=self._tier)
        elif key == "external_program_nprocs":
            self._nprocs = int(value)
        elif key == "external_program_memory":
            self._maxcore = int(value)
