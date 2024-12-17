from enum import Enum, auto
from typing import List, Dict, Union, Callable, Optional
from dataclasses import dataclass
import math
from constants import *
from neutrino_flux import *
from nr import *
from electroweak_er import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson as simps



@dataclass
class Target():
    AMU_TO_GEV = 0.931494

    def __init__(self, name: str, mass: float, charge: int, mass_frac= 1.0):
        self.name = name
        self.mass = mass
        self.charge = charge
        self.mass_frac = mass_frac

    def get_A(self):
        return self.mass
    
    def get_Z(self):
        return self.charge
    
    def get_m_GeV(self):
        return self.mass * self.AMU_TO_GEV
        

class CompositeNeutrinoFlux():
    def __init__(self):
        #super().__init__()
        self.components = []

    def clear(self):
        self.components.clear()

    def add_component(self, flux: NeutrinoFlux):
        self.components.append(flux)


    def get_total_flux_cm2s(self):
        total = 0
        for component in self.components:
            total += component.get_total_flux()

        return total

    def flavour_average(self, func, flavour):
        avg_for_flavour = 0
        for component in self.components:
            component.apply_oscillation()
            avg = component.flavour_average(func, flavour)
            avg_for_flavour += avg

        return avg_for_flavour


class NeutrinoRate:
    Components: List[str] = ["DSN", "Atmospheric", "8B", "HEP", "PP", "PEP", "CNO", "7Be", "7Be_PP_CNO"]

    def __init__(self, required_fluxes: str, interaction_type: InteractionType, target: Target):
        self.target = target
        self.f_interaction_type = interaction_type
        self.f_required_fluxes_str = required_fluxes
        self.f_flux = CompositeNeutrinoFlux()
        self.set_interaction_type(interaction_type)
        self.set_required_fluxes(required_fluxes)

    def set_required_fluxes(self, component: str):
        flux_map: Dict[str, List[str]] = {
            "DSN": ["dsnbflux_8", "dsnbflux_5", "dsnbflux_3"],
            "Atmospheric": ["AtmNu_e", "AtmNu_ebar", "AtmNu_mu", "AtmNu_mubar"],
            "8B": ["8B"],
            "HEP": ["hep"],
            "PEP": ["pep"],
            "PP": ["pp"],
            "CNO": ["13N", "15O", "17F"],
            "7Be": ["7Be_384.3keV", "7Be_861.3keV"],
            "7Be_PP_CNO": ["7Be_384.3keV", "7Be_861.3keV", "pp", "13N", "15O", "17F"]
        }
        required_neutrino_fluxes = flux_map.get(component, [])
        self.f_flux.clear()
        for key in required_neutrino_fluxes:
            print(key)
            flux = NeutrinoFlux(name=key, scaling=1.0, neutrino_flavour="e", oscillation_mode="solar_vac_sun",
                               # "solar_vac_sun"
                                ) 
            self.f_flux.add_component(flux)

    def set_interaction_type(self, interaction_type: InteractionType):
        self.f_interaction_type = interaction_type
        if interaction_type == InteractionType.COHERENT:
            self.f_cross_section = NeutrinoCrossSectionCoherentNR()
        elif interaction_type in {InteractionType.EW_FREE_ELECTRON, InteractionType.EW_STEPPING, InteractionType.EW_RRPA}:
            self.f_cross_section = NeutrinoCrossSectionElectroweakER(interaction_type) 

    def get_rate(self, recoil_keV: float) -> float:
        m_e = 0.511  # Electron mass in keV
        rate = 0.0

        for flavour in ["ElectronNeutrino", "MuonNeutrino", "TauNeutrino"]:
            rate_contrib = 0.0
            for nucleus in [self.target]:
                m_nuc = nucleus.get_m_GeV() * 1e6 #conversion to keV 

                if self.f_interaction_type == InteractionType.COHERENT:
                    if recoil_keV <= 0 or m_nuc <= 0:
                        raise ValueError("E_recoil and m_nuc must be positive values.")
                    term = 1 + (2 * m_nuc / recoil_keV)
                    E_nu_min = 0.5 * recoil_keV * (1 + np.sqrt(term))
    
                elif self.f_interaction_type in {InteractionType.EW_FREE_ELECTRON, InteractionType.EW_STEPPING, InteractionType.EW_RRPA}:
                    E_nu_min = 0.5 * (recoil_keV + math.sqrt(recoil_keV * (recoil_keV + 2 * m_e)))
                
                # averages cross section over neutrino flux, weighted based on flavour
                rate_function = lambda E_nu_keV: self.f_cross_section.dSigmadEr_cm2_keV(recoil_keV, E_nu_keV, nucleus, flavour) if E_nu_keV > E_nu_min else 0
                        
                dR_dE_r = self.f_flux.get_total_flux_cm2s() * max(0, self.f_flux.flavour_average(rate_function, flavour))
                factor = (5.61e35 / m_nuc) #TODO probably is wrong
  

                dR_dE_r *= (5.61e35 / m_nuc)
                rate_contrib += nucleus.mass_frac * dR_dE_r
            rate += rate_contrib
        return rate





# Main usage
def main():
    xe_target = Target("Xe", 131.29, 54)

    # Define interaction type (e.g., COHERENT)
    interaction_type = InteractionType.EW_RRPA

    # Required flux component (e.g., "8B" neutrinos from the sun)
    required_fluxes = "8B"

    # Initialise the neutrino rate calculation for xe target
    neutrino_rate = NeutrinoRate(required_fluxes, interaction_type, xe_target)

    # Calculate the neutrino scattering rate for a specific recoil energy (in keV)
    rates=[]
    recoil_energy_keV = np.linspace(0.1, 1, 10)  # Example recoil energy

    num_points = 100
    x_uniform = np.linspace(0.000001, 1, num_points)  # Uniformly distributed points in [0, 1]
    x_concentrated = x_uniform**2  # Squish points toward 0 (use x**n for more concentration)
    #recoil_energy_keV = x_concentrated * 80  # 
    for recoil_E in recoil_energy_keV:
        r = neutrino_rate.get_rate(recoil_E)
        rates.append(r)
    
    diff_rate = np.array(rates)* 365 * 60 * 60 * 24 * 1000 / (1e-38)

    plt.scatter(recoil_energy_keV, diff_rate)
    
    plt.xlabel('recoil energy [keV]')
    plt.ylabel(f'rate [events/tonne/year/keV]e^{-38} ')
    plt.savefig("example_recoil_spec.png")
    plt.show()

    diff_rate = np.array(rates)* 365 * 60 * 60 * 24 * 1000 #/ (1e-38)

    total_rate = simps(y=diff_rate, x=recoil_energy_keV)

    print(f"Total Event Rate (events/tonne/year): {total_rate}")
    # NOTE this DOESNT work - it gives you a very funky sin/cos graph

    # Output the computed rate
    print(f"Neutrino interaction rate for {recoil_energy_keV[0]} keV recoil on Xe: {rates[0]:.3e} events/kg/day")


if __name__ == "__main__":
    main()

