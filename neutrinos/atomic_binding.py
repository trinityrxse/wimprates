from typing import List
from enum import Enum
from constants import *
from propagate import *
import numpy as np
from bisect import bisect_left

class NeutrinoFlavour(Enum):
    Electron = 1
    Muon = 2
    Tau = 3

class AtomicBinding:
    """
    Class for applying stepping or RRPA atomic binding calculations for neutrino ER scattering.
    """

    def __init__(self, is_rrpa: bool, atomic_number: int):
        """
        Constructor for AtomicBinding.

        Args:
            is_rrpa (bool): Whether this is part of an RRPA cross-section.
            atomic_number (int): Atomic number of the element.
        """
        self.fAtomicNumber = atomic_number
        self.fExist = is_rrpa  # Whether RRPA files exist
        self.fEnergies_eType: List[float] = []
        self.fScaleFactor_eType: List[float] = []
        self.fEnergies_muTauType: List[float] = []
        self.fScaleFactor_muTauType: List[float] = []

        if is_rrpa and atomic_number == 54:
            scale_factor_eType = f"{DMCalcConstants.get_dmcalc_path()}/DataBase/NeutrinoEW_RRPA/Xe/Xe_eType.txt"
            scale_factor_muTauType = f"{DMCalcConstants.get_dmcalc_path()}/DataBase/NeutrinoEW_RRPA/Xe/Xe_muTauType.txt"

            try:
                with open(scale_factor_eType, 'r') as file_eType, open(scale_factor_muTauType, 'r') as file_muTauType:
                    self.fExist = True
                    
                    for line in file_eType:
                        energy, scaling = map(float, line.split())
                        self.fEnergies_eType.append(energy)
                        self.fScaleFactor_eType.append(scaling)
                    for line in file_muTauType:
                        energy, scaling = map(float, line.split())
                        self.fEnergies_muTauType.append(energy)
                        self.fScaleFactor_muTauType.append(scaling)
            

            except FileNotFoundError:
                print("Could not find Xe RRPA .txt files, using stepping approximation instead")
                self.fExist = False
            

    def active_electrons_stepping(self, recoil_keV: float) -> int:
        """
        Get the number of active electrons based on recoil energy and shell binding energies.

        Args:
            recoil_keV (float): Recoil energy (keV).

        Returns:
            int: Number of active electrons.
        """

        static_cache = {'cacheZ': None, 'shells': None}

        if static_cache['cacheZ'] != self.fAtomicNumber:
            static_cache['cacheZ'] = self.fAtomicNumber
            static_cache['shells'] = AtomicShells.create(self.fAtomicNumber)

        shells = static_cache['shells']
        atom_no_elec = shells.get_number_of_electrons()
        n_scattered_elec = 0
        atom_number_of_shells = shells.get_number_of_shells()
        atom_shell_bes = shells.get_shell_BEs()
        atom_shell_occupation = shells.get_shell_occupation()

        k_shell_energy = 0.001 * atom_shell_bes[0]  # Convert to keV

        if recoil_keV >= k_shell_energy:
            n_scattered_elec = atom_no_elec
        else:
            for j in range(atom_number_of_shells):
                shell_occupancy = atom_shell_occupation[j]
                binding_energy = 0.001 * atom_shell_bes[j]  # Convert to keV

                if recoil_keV >= binding_energy:
                    n_scattered_elec += shell_occupancy

        return n_scattered_elec

    def get_rrpa_scaling(self, neutrino_flavour: NeutrinoFlavour, recoil_keV: float) -> float:
        """
        Get the scaling factor for stepping --> RRPA.

        Args:
            neutrino_flavour (NeutrinoFlavour): Flavour of the neutrino.
            recoil_keV (float): Recoil energy (keV).

        Returns:
            float: Scaling factor.
        """
        rrpa_factor = 1.0  # Default scaling factor

        if self.fExist:

            if neutrino_flavour == 'e':
                energies, scale_factors = self.fEnergies_eType, self.fScaleFactor_eType
            elif neutrino_flavour in ('mu', 'tau'):
                energies, scale_factors = self.fEnergies_muTauType, self.fScaleFactor_muTauType
            else:
                return rrpa_factor  # Default for unknown flavour
            if energies[0] < recoil_keV < energies[-1]:
                idx_right = bisect_left(energies, recoil_keV)
                left_energy, right_energy = energies[idx_right - 1], energies[idx_right]
                left_scale, right_scale = scale_factors[idx_right - 1], scale_factors[idx_right]

                # Linear interpolation
                diff_scale_diff_energy = (right_scale - left_scale) / (right_energy - left_energy)
                rrpa_factor = left_scale + diff_scale_diff_energy * (recoil_keV - left_energy)
            elif recoil_keV >= energies[-1]:
                rrpa_factor = scale_factors[-1] # If recoil_keV exceeds all energies, return the last scaling factor
        
        return rrpa_factor

class AtomicShells:
    def __init__(self, Z, NumberOfShells, ShellBEs, ShellOccupation):
        self.fTAtomZ = Z
        self.fNumberOfShells = NumberOfShells
        self.fShellBEs = ShellBEs
        self.fShellOccupation = ShellOccupation

    @staticmethod
    def create(Z):
        # Placeholder logic; replace with actual initialization.
        if Z == 54:  # Example for Xenon
            NumberOfShells = 5
            ShellBEs = [34.56, 5.45, 3.4, 1.2, 0.54]  # Binding Energies in keV
            ShellOccupation = [2, 8, 18, 18, 8]  # Number of electrons per shell
            return AtomicShells(Z, NumberOfShells, ShellBEs, ShellOccupation)
        else:
            raise ValueError("AtomicShells data not available for given atomic number")

    def get_number_of_shells(self):
        return self.fNumberOfShells

    def get_number_of_electrons(self):
        return self.fTAtomZ

    def get_shell_BEs(self):
        return self.fShellBEs

    def get_shell_occupation(self):
        return self.fShellOccupation

# Example usage
if __name__ == "__main__":
    binding = AtomicBinding(is_rrpa=True, atomic_number=54)
    active_electrons = binding.active_electrons_stepping(.001)  # Example recoil energy in keV
    scaling = binding.get_rrpa_scaling("ElectronNeutrino", .001)

    print(f"Active Electrons: {active_electrons}")
    print(f"RRPA Scaling Factor: {scaling}")
