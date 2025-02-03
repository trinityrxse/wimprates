from enum import Enum
from typing import Optional
from atomic_binding import *
from neutrino_flux import *

class ElectroweakAtomicBinding(Enum):
    """
    Electroweak binding types.
    - FreeElectron
    - Stepping
    - RRPA
    """
    FREE_ELECTRON = "FreeElectron"
    STEPPING = "Stepping"
    RRPA = "RRPA"


class NeutrinoCrossSectionElectroweakER(VNeutrinoCrossSection):
    """
    Neutrino electroweak cross-section class.
    """
    #TODO fix this so it works with antineutrinos

    def __init__(self, binding_flag: InteractionType):
        super().__init__()
        self.fBindingFlag = binding_flag
        self.fBinding: Optional[AtomicBinding] = None
        self.fCoupling_VAsumsq = 1  # (C_V + C_A)^2
        self.fCoupling_VAdiffsq = 1  # (C_V - C_A)^2
        self.fCoupling_AVsqdiff = 1  # C_A^2 - C_V^2
        self.fIsElectronNeutrino = True
        self.fIsAntiNeutrino = False


    def dSigmadEr_cm2_keV(self, E_recoil_keV: float, E_neutrino_MeV: float, nucleus, flavour):
        self.SetCouplings(flavour)

        A_nuc = nucleus.get_A()  # mass number
        Z_nuc = nucleus.get_Z()  # atomic number
        m_nuc_keV = nucleus.get_m_GeV() * 1e6  # keV


        if self.fBindingFlag == InteractionType.EW_RRPA:
            self.SetBinding(is_rrpa=bool(True), Z_nuc=Z_nuc)
        else:
            self.SetBinding(is_rrpa=bool(False), Z_nuc=Z_nuc)

        # Constants
        Gf = DMCalcConstants.Gf
        m_e = DMCalcConstants.m_e_keV

        recoil2 = (1 - E_recoil_keV / ((E_neutrino_MeV * 1e3)) ** 2)
        prefactor = (Gf ** 2 * m_e) / (2 * math.pi)

        if self.fIsAntiNeutrino:
            dxsecdEr = prefactor * (
                self.fCoupling_VAsumsq * recoil2 +
                self.fCoupling_VAdiffsq +
                self.fCoupling_AVsqdiff * m_e * E_recoil_keV / ((E_neutrino_MeV * 1e3) ** 2)
            )
        else:
            dxsecdEr = prefactor * (
                self.fCoupling_VAsumsq +
                self.fCoupling_VAdiffsq * recoil2 +
                self.fCoupling_AVsqdiff * m_e * E_recoil_keV / ((E_neutrino_MeV * 1e3)  ** 2)
            )

        if self.fBindingFlag== InteractionType.EW_FREE_ELECTRON:
            dxsecdEr *= Z_nuc 
        elif self.fBindingFlag == InteractionType.EW_STEPPING:
            acte = self.fBinding.active_electrons_stepping(E_recoil_keV)
            dxsecdEr *= acte 
        elif self.fBindingFlag == InteractionType.EW_RRPA:
            acte = self.fBinding.active_electrons_stepping(E_recoil_keV)
            dxsecdEr *= acte 
            RRPA_factor = 1.0
            if not self.fIsAntiNeutrino:
                if self.fIsElectronNeutrino:
                    RRPA_factor = self.fBinding.get_rrpa_scaling(NeutrinoFlavour.ElectronNeutrino, E_recoil_keV)
                else:
                    RRPA_factor = self.fBinding.get_rrpa_scaling(NeutrinoFlavour.MuonNeutrino, E_recoil_keV)
        
            dxsecdEr /= RRPA_factor

        dxsecdEr = dxsecdEr * 3.88e-16 # cm^2/keV #why might this be different e-13...

        return dxsecdEr

    def SetCouplings(self, neutrino_flavour):
        C_V = 2 * DMCalcConstants.sin_sq_theta_W - 0.5
        C_A = -0.5
        C_V_e = 2 * DMCalcConstants.sin_sq_theta_W - 0.5 + 1.0
        C_A_e = -0.5 + 1.0

        if neutrino_flavour == "ElectronNeutrino":
            self.fCoupling_VAsumsq = (C_V_e + C_A_e) ** 2
            self.fCoupling_VAdiffsq = (C_V_e - C_A_e) ** 2
            self.fCoupling_AVsqdiff = C_A_e ** 2 - C_V_e ** 2
            self.fIsElectronNeutrino = True
            self.fIsAntiNeutrino = False
        elif neutrino_flavour in ("MuonNeutrino", "TauNeutrino"):
            self.fCoupling_VAsumsq = (C_V + C_A) ** 2
            self.fCoupling_VAdiffsq = (C_V - C_A) ** 2
            self.fCoupling_AVsqdiff = C_A ** 2 - C_V ** 2
            self.fIsElectronNeutrino = False
            self.fIsAntiNeutrino = False
        elif neutrino_flavour == "AntiElectronNeutrino":
            self.fCoupling_VAsumsq = (C_V_e + C_A_e) ** 2
            self.fCoupling_VAdiffsq = (C_V_e - C_A_e) ** 2
            self.fCoupling_AVsqdiff = C_A_e ** 2 - C_V_e ** 2
            self.fIsElectronNeutrino = True
            self.fIsAntiNeutrino = True
        elif neutrino_flavour in ("AntiMuonNeutrino","AntiTauNeutrino"):
            self.fCoupling_VAsumsq = (C_V + C_A) ** 2
            self.fCoupling_VAdiffsq = (C_V - C_A) ** 2
            self.fCoupling_AVsqdiff = C_A ** 2 - C_V ** 2
            self.fIsElectronNeutrino = False
            self.fIsAntiNeutrino = True


    def SetBinding(self, is_rrpa: bool, Z_nuc: int):
        """
        Instantiate an atomic binding class instance.
        :param is_rrpa: whether this is an RRPA cross-section
        :param Z_nuc: atomic number of the nucleus
        """
        self.fBinding = AtomicBinding(is_rrpa, Z_nuc)
