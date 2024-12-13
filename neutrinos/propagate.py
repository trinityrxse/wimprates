import numpy as np
from typing import Optional
from numpy import array, complex128
from sun_density import SunDensity
from scipy.linalg import expm

class Propagate:
    """
    Solar neutrino propagation class.
    """
    fCoeff1 = 1.27 * 4  # Coefficient to get units right.
    fCoeff2 = 6.42e-28  # Coefficient to get units right.

    def __init__(
        self,
        density_file: str,
        energy_GeV: float,
        deltaMsq21: float,
        deltaMsq32: float,
        theta12: float,
        theta13: float,
        theta23: float,
        delta: float,
        averaging: int,
    ):
        self.fSun = SunDensity(density_file)  # Placeholder instance for SunDensity class
        self.fEnergy_GeV = energy_GeV
        self.fDeltaMsq21 = deltaMsq21
        self.fDeltaMsq32 = deltaMsq32
        self.fTheta12 = theta12
        self.fTheta13 = theta13
        self.fTheta23 = theta23
        self.fDelta = delta
        self.fAveraging = averaging

        # Initialize the Hamiltonian matrices and neutrino state vector
        self.fHamiltonianMass = np.zeros((3, 3), dtype=complex128)
        self.fHamiltonian = np.zeros((3, 3), dtype=complex128)
        self.fPsi = np.zeros(3, dtype=complex128)

        self.set_hamiltonian_mass()

    def change_params(
        self,
        energy_GeV: float,
        deltaMsq21: float,
        deltaMsq32: float,
        theta12: float,
        theta13: float,
        theta23: float,
        delta: float,
    ):
        """
        Change oscillation parameters.
        """
        self.fEnergy_GeV = energy_GeV
        self.fDeltaMsq21 = deltaMsq21
        self.fDeltaMsq32 = deltaMsq32
        self.fTheta12 = theta12
        self.fTheta13 = theta13
        self.fTheta23 = theta23
        self.fDelta = delta

        self.set_hamiltonian_mass()

    def set_hamiltonian_mass(self):
        """
        Calculate the mass part of the Hamiltonian.
        Placeholder for the actual implementation.
        """
        s12 = np.sin(self.fTheta12)
        c12 = np.cos(self.fTheta12)
        s13 = np.sin(self.fTheta13)
        c13 = np.cos(self.fTheta13)
        s23 = np.sin(self.fTheta23)
        c23 = np.cos(self.fTheta23)
        deltaExp_minus = np.exp(-1j * self.fDelta)
        deltaExp_plus = np.exp(1j * self.fDelta)

        PMNS = np.array([
            [c12 * c13, s12 * c13, s13 * deltaExp_minus],
            [-s12 * c23 - c12 * s23 * s13 * deltaExp_plus, c12 * c23 - s12 * s23 * s13 * deltaExp_plus, s23 * c13],
            [s12 * s23 - c12 * c23 * s13 * deltaExp_plus, -c12 * s23 - s12 * c23 * s13 * deltaExp_plus, c23 * c13]
        ], dtype=np.complex128)

        PMNS_dagger = PMNS.conj().T

        HamiltonianMass_massBasis = np.zeros((3, 3), dtype=np.complex128)
        HamiltonianMass_massBasis[1, 1] = self.fDeltaMsq21 * 0.5 * self.fCoeff1 / self.fEnergy_GeV
        HamiltonianMass_massBasis[2, 2] = self.fDeltaMsq32 * 0.5 * self.fCoeff1 / self.fEnergy_GeV

        temp_storage = PMNS_dagger @ HamiltonianMass_massBasis
        self.fHamiltonianMass = PMNS @ temp_storage

    def get_transition_prob(
        self, min_path_km: float, nu_start: int, nu_end: int, outwards: bool
    ) -> float:
        """
        Get neutrino transition probability.

        :param min_path_km: Neutrino starting location in the sun (radially) in km.
        :param nu_start: Starting flavour (1=electron, 2=muon, 3=tau).
        :param nu_end: Final flavour (1=electron, 2=muon, 3=tau).
        :param outwards: Whether the neutrino travels radially outwards or first inwards through the solar core.
        :return: Transition probability from nu_start->nu_end.
        """
        shells_ahead = self.fSun.get_shells_ahead(min_path_km)
        shells = self.fSun.get_num_shells()
        start_shell = shells - shells_ahead
        self.fPsi.fill(0)
        self.fPsi[nu_start - 1] = 1
        probabilities = []

        if nu_start == nu_end:
            probabilities.append(1.0)

        if not outwards:
            for i in range(start_shell - 1, -1, -1):
                self._propagate_shell(i, nu_end, probabilities)

            start_shell = 0

        for i in range(start_shell, shells):
            self._propagate_shell(i, nu_end, probabilities)

        vacuum_wavelength = 2.47 * self.fEnergy_GeV / self.fDeltaMsq21
        vacuum_distance = 50 * vacuum_wavelength / self.fAveraging

        for _ in range(self.fAveraging):
            transition = expm(-1j * self.fHamiltonianMass * vacuum_distance)
            self.fPsi = transition @ self.fPsi
            probabilities.append(abs(self.fPsi[nu_end - 1]) ** 2)

        probability_averaged = np.mean(probabilities[-self.fAveraging:])
        return probability_averaged
    
    def _propagate_shell(self, i, nu_end, probabilities):
        self.fHamiltonian = self.fHamiltonianMass.copy()
        self.fHamiltonian[0, 0] += self.fCoeff2 * self.fSun.GetDensityInShell(i)
        distance = self.fSun.GetDistanceAcrossShell(i)
        transition = expm(-1j * self.fHamiltonian * distance)
        self.fPsi = transition @ self.fPsi
        probabilities.append(abs(self.fPsi[nu_end - 1]) ** 2)


        
# NOTE Example usage:
# propagate = Propagate("electron_density.txt", 1.0, 7.4e-5, 2.5e-3, 33.44, 8.57, 45.0, 0.0, 10)
# propagate.set_hamiltonian_mass()
# prob = propagate.get_transition_prob(10, 1, 1, True)


