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

    def dSigmadEr_cm2_keV(self, E_recoil: float, E_neutrino: float, nucleus: Nucleus) -> float:
        """
        Calculate the cross-section.
        :param E_recoil: nuclear recoil energy in keV
        :param E_neutrino: incident neutrino energy in keV
        :param nucleus: target nucleus
        :return: value of the cross-section in cm^2.keV^-1
        """
        # Placeholder for actual cross-section calculation

        A_nuc = nucleus.GetA() #mass number
        Z_nuc = nucleus.GetZ() #atomic number
        m_nuc = nucleus.GetMGeV() * 1e6 # actual mass, conversion in keV

        Gf = DMCalcConstants.Gf

        Fsquared = wr.helm_form_factor_squared(E_recoil, A_nuc)

        x = E_recoil / E_neutrino
        y = m_nuc / E_neutrino
        dxsecdEr = ((Gf**2) / (2 * np.pi)) * (self.fCoupling_v_proton * Z_nuc + 
                                            self.fCoupling_v_neutron * (A_nuc - Z_nuc)) * (self.fCoupling_v_proton * Z_nuc + 
                                            self.fCoupling_v_neutron * (A_nuc - Z_nuc)) * m_nuc * (1 + (1 - x) * (1 - x) - x * y) * Fsquared

        dxsecdEr = dxsecdEr * 3.88e-16 # cm^2/keV

        return np.fmax(dxsecdEr, 0)


        cross_section = (self.fCoupling_v_proton + self.fCoupling_v_neutron) * E_recoil * E_neutrino / nucleus.mass
        return cross_section

    def SetCouplings(self, neutrinoFlavour: NeutrinoFlavour):
        """
        Set the couplings.
        :param neutrinoFlavour: neutrino flavour enum
        """
        # Placeholder for setting couplings based on neutrino flavour
        if neutrinoFlavour == NeutrinoFlavour.ELECTRON:
            self.fCoupling_v_proton = 0.03824555057133305
            self.fCoupling_v_neutron = -0.511669383346544
        elif neutrinoFlavour == NeutrinoFlavour.MUON:
            self.fCoupling_v_proton = 0.029989302410145635
            self.fCoupling_v_neutron = -0.5116693833465439
        elif neutrinoFlavour == NeutrinoFlavour.TAU:
            self.fCoupling_v_proton = 0.025618721492906504
            self.fCoupling_v_neutron = -0.511669383346544

