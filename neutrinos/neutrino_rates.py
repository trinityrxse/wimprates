from enum import Enum, auto
from typing import List, Dict, Union, Callable, Optional
from dataclasses import dataclass
import math
from constants import *
from neutrino_flux import *

class InteractionType(Enum):
    COHERENT = auto()
    EW_FREE_ELECTRON = auto()
    EW_STEPPING = auto()
    EW_RRPA = auto()

class RecoilType(Enum):
    NR = auto()
    ER = auto()

@dataclass
class Target:
    Z: int
    mass_fraction: float

    @staticmethod
    def create(material: str) -> Optional['Target']:
        # Placeholder: Create target based on name
       
        return Target(Z=ATOMIC_WEIGHT[material], mass_fraction=1.0)  # Example values


class NeutrinoFlux:
    def get_total_flux_icm2s(self):
        # Placeholder for total flux computation
        return 1.0

    def flavor_average(self, rate_function: Callable[[float], float], flavor: str) -> float:
        # Placeholder for averaging over flavors
        return 1.0

class CompositeNeutrinoFlux:
    def __init__(self):
        self.components = []

    def clear(self):
        self.components.clear()

    def add_component(self, flux: NeutrinoFlux):
        self.components.append(flux)


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
            flux = NeutrinoFlux()  # Placeholder for database access
            self.f_flux.add_component(flux)

    def set_interaction_type(self, interaction_type: InteractionType):
        self.f_interaction_type = interaction_type
        if interaction_type == InteractionType.COHERENT:
            self.f_cross_section = VNeutrinoCrossSection()  # Placeholder
        elif interaction_type in {InteractionType.EW_FREE_ELECTRON, InteractionType.EW_STEPPING, InteractionType.EW_RRPA}:
            self.f_cross_section = VNeutrinoCrossSection()  # Placeholder

    def get_rate(self, recoil_keV: float) -> float:
        m_e = 0.511  # Electron mass in keV
        rate = 0.0
        for flavor in ["ElectronNeutrino", "MuonNeutrino", "TauNeutrino"]:
            rate_contrib = 0.0
            for nucleus in [self.target]:
                m_nuc = nucleus.Z * 1e6  # Placeholder for nucleus mass
                E_nu_min = 0.0
                if self.f_interaction_type == InteractionType.COHERENT:
                    E_nu_min = math.sqrt(m_nuc * recoil_keV / 2)
                elif self.f_interaction_type in {InteractionType.EW_FREE_ELECTRON, InteractionType.EW_STEPPING, InteractionType.EW_RRPA}:
                    E_nu_min = 0.5 * (recoil_keV + math.sqrt(recoil_keV * (recoil_keV + 2 * m_e)))

                rate_function = lambda E_nu_keV: self.f_cross_section.d_sigma_d_erg_cm2_keV(recoil_keV, E_nu_keV, nucleus) if E_nu_keV >= E_nu_min else 0.0
                dR_dE_r = self.f_flux.get_total_flux_icm2s() * max(0, self.f_flux.flavor_average(rate_function, flavor))
                dR_dE_r *= (5.61e35 / m_nuc)
                rate_contrib += nucleus.mass_fraction * dR_dE_r
            rate += rate_contrib
        return rate
