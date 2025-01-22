from typing import Optional
from atomic_binding import *
from neutrino_flux import *
from formfactorcheck import *
import numpy as np
import wimprates as wr
from scipy.special import jn

class NeutrinoCrossSectionCoherentNR(VNeutrinoCrossSection):
    """
    Neutrino coherent scattering cross-section class.
    """

    def __init__(self):
        super().__init__()
        self.fCoupling_v_proton = 1  # Vector coupling to proton (radiative corrections)
        self.fCoupling_v_neutron = 1  # Vector coupling to neutron (radiative corrections)

    def dSigmadEr_cm2_keV(self, E_recoil_keV: float, E_neutrino_MeV: float, nucleus, flavour) -> float:
        """
        Calculate the cross-section.
        :param E_recoil_keV: nuclear recoil energy in keV
        :param E_neutrino_MeV: incident neutrino energy in MeV
        :param nucleus: target nucleus
        :return: value of the cross-section in cm^2.keV^-1
        """

        #TODO check with Rob do we need to set couplings and have oscillations for antineutrinos
        self.set_couplings(flavour)
        #print(E_recoil, E_neutrino)
        A_nuc = nucleus.get_A()  # mass number
        Z_nuc = nucleus.get_Z()  # atomic number
        m_nuc_keV = nucleus.get_m_GeV() * 1e6  # keV

        Gf = DMCalcConstants.Gf  # Fermi constant in keV^-2

        # Ensure helm_form_factor_squared returns a dimensionless result
        Fsquared = helm_form_factor(E_recoil_keV, A_nuc, nucleus.get_m_GeV()) #Er_keV, A, mass_GeV
        #print(Fsquared, 'fsq')

        # Temporary debug fix for Fsquared
        if Fsquared is None or Fsquared <= 0:  # Handle any invalid values
            Fsquared = 0  # Default value (fix this function later)

        x = E_recoil_keV / (E_neutrino_MeV * 1e3)
        y = m_nuc_keV / (E_neutrino_MeV * 1e3)
        dxsecdEr = Gf * Gf / (2. * np.pi) * (self.fCoupling_v_proton * Z_nuc + self.fCoupling_v_neutron * (A_nuc - Z_nuc)) * (self.fCoupling_v_proton * Z_nuc + self.fCoupling_v_neutron * (A_nuc - Z_nuc)) * m_nuc_keV * (1 + (1 - x) * (1 - x) - x * y) * Fsquared

        dxsecdEr = dxsecdEr * 3.88e-16 # cm^2/keV

        return dxsecdEr


    def set_couplings(self, neutrinoFlavour: str):
        """
        Set the couplings.
        :param neutrinoFlavour: neutrino flavour enum
        """
        if neutrinoFlavour == "ElectronNeutrino":
            self.fCoupling_v_proton = 0.03824555057133305
            self.fCoupling_v_neutron = -0.511669383346544
        elif neutrinoFlavour == "MuonNeutrino":
            self.fCoupling_v_proton = 0.029989302410145635
            self.fCoupling_v_neutron = -0.5116693833465439
        elif neutrinoFlavour == "TauNeutrino":
            self.fCoupling_v_proton = 0.025618721492906504
            self.fCoupling_v_neutron = -0.511669383346544

    def form_factor_plot(self, nucleus):
        a = nucleus.get_A()
        mass_GeV = nucleus.get_m_GeV()

        erec_keV = np.logspace(-1, 3, 1000)
        data = []
        for erec in erec_keV:
            form_factor_squared = helm_form_factor(erec, a, mass_GeV)**2

            data.append([erec, form_factor_squared])
        
        data = np.array(data)
        plt.scatter(data[:,0], data[:,1])
        plt.yscale("log") 
        plt.xlabel("T [keV]")
        plt.ylabel("F^2")
        plt.savefig('ff.png')
        plt.show()
    
            

    
"""    def helm_form_factor_plot(self, erec_keV, nucleus):

        anucl = nucleus.get_A() 

        c = 1.23 * anucl**(1/3) - 0.60  # Effective nuclear radius parameter (fm)
        a = 0.52  # Diffuseness parameter (fm)
        s = 0.9   # Skin thickness parameter (fm)

        # Compute root-mean-square nuclear radius squared (in fm^2)
        rn_sq = c**2 + (7.0 / 3.0) * (np.pi**2) * a**2 - 5 * s**2
        rn_fm = np.sqrt(rn_sq)  # Root-mean-square radius in fm
  
        m_nucleus_keV = nucleus.get_m_GeV() * 1e6
    
        hbarc_keV_fm = 1.97327e4
        erec_keV = np.logspace(0, 3, 1000)
        data = []
        for erec in erec_keV:


            # Momentum transfer q (in keV/c)
            #q_keV = np.sqrt(erec**2 + 2 * m_nucleus_keV * erec)  # Momentum transfer in keV/c
            #low energy approximation
            q_keV = np.sqrt(2 * m_nucleus_keV * erec)


            # Convert q from keV/c to fm^-1 (momentum transfer in inverse femtometers)
            q_fm_inverse = q_keV / hbarc_keV_fm

            # Spherical Bessel function j1(q * R_n)
            j1_qr = jn(1, q_fm_inverse * rn_fm)

            # Exponential term e^(-0.5 * (q * s)^2)
            exp_term = np.exp(-0.5 * (q_fm_inverse * s)**2)

            # Helm form factor formula
            form_factor = (3 * j1_qr) * (exp_term / (q_fm_inverse * rn_fm))

            # Helm form factor squared (already dimensionless)
            form_factor_squared = form_factor**2

            data.append([erec, form_factor**2])


        data = np.array(data)
        plt.scatter(data[:,0], data[:,1])
        plt.yscale("log") 
        plt.xlabel("T [keV]")
        plt.ylabel("F^2")
        plt.savefig('ff.png')
        plt.show()







def helm_form_factor_squared2(anucl, erec_keV, m_nucleus_keV):
    # Constants in keV·fm
    hbarc_keV_fm = 1.97327e4  # hbar*c in keV·fm (converted from GeV·fm)

    # Helm model parameters (fm)
    c = 1.23 * anucl**(1/3) - 0.60  # Effective nuclear radius parameter (fm)
    a = 0.52  # Diffuseness parameter (fm)
    s = 0.9   # Skin thickness parameter (fm)

    # Compute root-mean-square nuclear radius squared (fm^2)
    rn_sq = c**2 + (7.0 / 3.0) * (np.pi**2) * a**2 - 5 * s**2
    rn_fm = np.sqrt(rn_sq)  # Root-mean-square radius in fm

    # Momentum transfer q (in keV/c)
    q_keV = np.sqrt(erec_keV**2 + 2 * m_nucleus_keV * erec_keV)  # Momentum transfer in keV/c

    # Convert q from keV/c to fm^-1 (momentum transfer in inverse femtometers)
    q_fm_inverse = q_keV / hbarc_keV_fm

    # Spherical Bessel function j1(q * R_n)
    j1_qr = jn(1, q_fm_inverse * rn_fm)

    # Exponential term e^(-0.5 * (q * s)^2)
    exp_term = np.exp(-0.5 * (q_fm_inverse * s)**2)

    # Helm form factor formula
    form_factor = (3 * j1_qr) * (exp_term / (q_fm_inverse * rn_fm))

    # Helm form factor squared (already dimensionless)
    form_factor_squared = form_factor**2

    return form_factor_squared


def spherical_bessel_j1(x):
    #Spherical Bessel function j1 according to Wolfram Alpha
    return np.sin(x) / x**2 - np.cos(x) / x

"""