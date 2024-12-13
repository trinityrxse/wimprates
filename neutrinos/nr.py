from typing import Optional
from atomic_binding import *
from neutrino_flux import *
import numpy as np
import wimprates as wr

class NeutrinoCrossSectionCoherentNR(VNeutrinoCrossSection):
    """
    Neutrino coherent scattering cross-section class.
    """

    def __init__(self):
        super().__init__()
        self.fCoupling_v_proton = 1  # Vector coupling to proton (radiative corrections)
        self.fCoupling_v_neutron = 1  # Vector coupling to neutron (radiative corrections)

    def dSigmadEr_cm2_keV(self, E_recoil: float, E_neutrino: float, nucleus, flavour) -> float:
        """
        Calculate the cross-section.
        :param E_recoil: nuclear recoil energy in keV
        :param E_neutrino: incident neutrino energy in keV
        :param nucleus: target nucleus
        :return: value of the cross-section in cm^2.keV^-1
        """
        # Placeholder for actual cross-section calculation

        self.set_couplings(flavour)

        A_nuc = nucleus.get_A()  # mass number
        Z_nuc = nucleus.get_Z()  # atomic number
        m_nuc = nucleus.get_m_GeV() * 1e6  # actual mass in keV

        Gf = DMCalcConstants.Gf  # Fermi constant in keV^-2

        # Ensure helm_form_factor_squared returns a dimensionless result
        Fsquared = wr.helm_form_factor_squared(E_recoil, A_nuc)

        # Temporary debug fix for Fsquared
        if Fsquared is None or Fsquared < 0:  # Handle any invalid values
            Fsquared = 1  # Default value (fix this function later)

        x = E_recoil / E_neutrino if E_neutrino > 0 else 0 # Ratio of recoil to neutrino energy, dimensionless
        y = m_nuc / E_neutrino if E_neutrino > 0 else 0  # Ratio of nucleus mass to neutrino energy, dimensionless

        # Differential cross-section calculation
        dxsecdEr = ((Gf**2) / (2 * np.pi)) * (
            self.fCoupling_v_proton * Z_nuc + self.fCoupling_v_neutron * (A_nuc - Z_nuc)
        ) ** 2 * x
        #m_nuc*(1 + (1-x)**2) - (x*y)) * Fsquared
        #TODO fix this - make sure you have the right formula
        """        print(((Gf**2) / (2 * np.pi)))
        print((self.fCoupling_v_proton * Z_nuc + self.fCoupling_v_neutron * (A_nuc - Z_nuc))**2)
        print(m_nuc * (1 + ((1 - x)** 2) - (x * y)) * Fsquared)
        print((m_nuc*(1 + (1-x)**2)) - (x*y))
        print(x*y)"""

        # Convert to cm^2/keV
        dxsecdEr = dxsecdEr * 3.88e-16  # Conversion factor from natural units to cm^2/keV

        # Ensure non-negative values
        return np.fmax(dxsecdEr, 0)


    def set_couplings(self, neutrinoFlavour: str):
        """
        Set the couplings.
        :param neutrinoFlavour: neutrino flavour enum
        """
        # Placeholder for setting couplings based on neutrino flavour
        if neutrinoFlavour == "ElectronNeutrino":
            self.fCoupling_v_proton = 0.03824555057133305
            self.fCoupling_v_neutron = -0.511669383346544
        elif neutrinoFlavour == "MuonNeutrino":
            self.fCoupling_v_proton = 0.029989302410145635
            self.fCoupling_v_neutron = -0.5116693833465439
        elif neutrinoFlavour == "TauNeutrino":
            self.fCoupling_v_proton = 0.025618721492906504
            self.fCoupling_v_neutron = -0.511669383346544

