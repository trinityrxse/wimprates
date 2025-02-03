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


    def get_total_flux_cm2s(self, E_min: float = None, E_max: float = None):
        #for if the flux we want has more than one component (more than one file)
        total = 0
        for component in self.components:
            total += component.get_total_flux()
            
             # Result is in /s/cm^2
            #print('tot flux', component.get_total_flux(E_min, E_max), component.name)

        return total #this is the flux for all of the components loaded

    def flavour_average(self, func, flavour):
        avg_for_flavour = 0
        for component in self.components:
            #component.oscillation_mode == OscillationMode.NoneMode
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

        scaling = {
                "pp": {"flux": 5.98e10, "flavour": "e", "source": "solar_vac_sun"
                      #"solar_matter_sun"
                       },
                "pep": {"flux": 1.44e8, "flavour": "e", "source": "solar_vac_sun"
                       # "solar_matter_sun"
                        },
                "hep": {"flux": 7.98e3, "flavour": "e", "source": "solar_vac_sun"
                       # "solar_matter_sun"
                        },
                "7Be_384.3keV": {"flux": 6.44e8, "flavour": "e", "source": "solar_vac_sun"
                              #   "solar_matter_sun"
                                 },
                "7Be_861.3keV": {"flux": 4.35e9, "flavour": "e", "source": "solar_vac_sun"
                             #    "solar_matter_sun"
                                 },
                "8B": {"flux": 5.25e6, "flavour": "e", "source": "solar_vac_sun"
                     #  "solar_matter_sun"
                       },
                "13N": {"flux": 2.78e8, "flavour": "e", "source": "solar_vac_sun"
                      #  "solar_matter_sun"
                        },
                "15O": {"flux": 2.05e8, "flavour": "e", "source": "solar_vac_sun"
                     #   "solar_matter_sun"
                        },
                "17F": {"flux": 5.29e6, "flavour": "e", "source": "solar_vac_sun"
                    #    "solar_matter_sun"
                        },
                "dsnbflux_8": {"flux": 17.0, "flavour": "mu", "source": "none"},
                "dsnbflux_5": {"flux": 27.2, "flavour": "e_anti", "source": "none"},
                "dsnbflux_3": {"flux": 45.4, "flavour": "e", "source": "none"},
                "AtmNu_e": {"flux": 1.0, "flavour": "e", "source": "none"},
                "AtmNu_ebar": {"flux": 1.0, "flavour": "e_anti", "source": "none"},
                "AtmNu_mu": {"flux": 1.0, "flavour": "mu", "source": "none"},
                "AtmNu_mubar": {"flux": 1.0, "flavour": "mu_anti", "source": "none"}
            }
        required_neutrino_fluxes = flux_map.get(component, [])
        self.f_flux.clear()
        for key in required_neutrino_fluxes:
            flux = NeutrinoFlux(name=key, scaling=scaling[key]['flux'], neutrino_flavour=[scaling[key]['flavour']], oscillation_mode=scaling[key]['source'],
                                    # "solar_vac_sun"
                                        ) 
            #flux.apply_oscillation()
            self.f_flux.add_component(flux)

    def set_interaction_type(self, interaction_type: InteractionType):
        self.f_interaction_type = interaction_type
        if interaction_type == InteractionType.COHERENT:
            self.f_cross_section = NeutrinoCrossSectionCoherentNR()
        elif interaction_type in {InteractionType.EW_FREE_ELECTRON, InteractionType.EW_STEPPING, InteractionType.EW_RRPA}:
            self.f_cross_section = NeutrinoCrossSectionElectroweakER(interaction_type) 

    def get_rate(self, recoil_keV: float) -> float:
        m_e = DMCalcConstants.m_e_keV  # Electron mass in keV
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

                def rate_function(E_nu_MeV):
                    if (E_nu_MeV * 1e3) > E_nu_min:
                        return self.f_cross_section.dSigmadEr_cm2_keV(recoil_keV, E_nu_MeV, nucleus, flavour)
                    else:
                        return 0

                dR_dE_r = max(0, self.f_flux.flavour_average(rate_function, flavour))
                dR_dE_r *= 5.61e35 / m_nuc
                dR_dE_r = dR_dE_r * 365. * 24. * 3600. # evts/keV/tonne/yr
                rate_contrib += nucleus.mass_frac * dR_dE_r
            
            rate += rate_contrib

        return rate
    
    