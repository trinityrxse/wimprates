import math
from enum import Enum, auto
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Tuple, Union
import numericalunits as nu
from lookup_solar_MSW_table import SolarMSWLookupTable
import os
from constants import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson as simps

class InteractionType(Enum):
    COHERENT = auto()
    EW_FREE_ELECTRON = auto()
    EW_STEPPING = auto()
    EW_RRPA = auto()

class RecoilType(Enum):
    NR = auto()
    ER = auto()

class NeutrinoFlavour:
    ElectronNeutrino = "e"
    MuonNeutrino = "mu"
    TauNeutrino = "tau"
    AntiElectronNeutrino = "e_anti"
    AntiMuonNeutrino = "mu_anti"
    AntiTauNeutrino = "tau_anti"

class OscillationMode:
    NoneMode = "none"
    VacuumSunProduction = "solar_vac_sun"
    MatterSunProduction = "solar_matter_sun"

# Neutrino flux modes
class FluxMode:
    Continuous = 'continuous'
    Line = 'line'



class VNeutrinoCrossSection:
    """
    Abstract base class for neutrino cross-section calculations.
    """
    def dSigmadEr_cm2_keV(self, recoil_keV: float, neutrino_keV: float, nucleus) -> float:
        raise NotImplementedError

    def SetCouplings(self, neutrinoFlavour: NeutrinoFlavour):
        raise NotImplementedError

    def SetBinding(self, isRRPA: bool, Z_nuc: int):
        pass

class NeutrinoFlux:
    def __init__(self, name, scaling: float,
                 neutrino_flavour: str, oscillation_mode: str):
        self.name = name
        self.scaling = scaling
        self.mode = FluxMode.Continuous
        self.flavours = defaultdict(list)
        self.flavour_map = {
            "e": NeutrinoFlavour.ElectronNeutrino,
            "mu": NeutrinoFlavour.MuonNeutrino,
            "tau": NeutrinoFlavour.TauNeutrino,
            "e_anti": NeutrinoFlavour.AntiElectronNeutrino,
            "mu_anti": NeutrinoFlavour.AntiMuonNeutrino,
            "tau_anti": NeutrinoFlavour.AntiTauNeutrino,
        }
        self.oscillation_map = {
            "none": OscillationMode.NoneMode,
            "solar_vac_sun": OscillationMode.VacuumSunProduction,
            "solar_matter_sun": OscillationMode.MatterSunProduction,
        }
        self.oscillation_mode = self.oscillation_map[oscillation_mode]
        for neutrino_flav in neutrino_flavour:
            self.add_neutrino_flavour(self.flavour_map[neutrino_flav])


        self.pdf_data = self.pdf_data()

    def add_neutrino_flavour(self, flavour: str, fractions: Union[List[float], None] = None):
        if fractions is None:
            fractions = [1.0]
        self.flavours[flavour].extend(fractions)

    def apply_oscillation(self):
        theta_12 = math.asin(math.sqrt(DMCalcConstants.sin_sq_theta_12))
        theta_13 = math.asin(math.sqrt(DMCalcConstants.sin_sq_theta_13))
        theta_23 = math.asin(math.sqrt(DMCalcConstants.sin_sq_theta_23))

        npoints = 1 if self.mode == FluxMode.Line else len(self.pdf_data) - 1
        if self.oscillation_mode == "none":
            keys = list(self.flavours.keys())
            self.flavours.clear()    
            self.add_neutrino_flavour(keys[0], [1.0] * npoints)

        elif self.oscillation_mode == "solar_vac_sun":
            p_ee = 1 - 0.5 * math.pow(math.sin(2 * theta_12), 2) * math.pow(math.cos(theta_13), 4) - 0.5 * math.pow(math.sin(2 * theta_13), 2)
            p_emu = -0.5 * math.sin(2 * theta_12) * math.pow(math.cos(theta_13), 2) * (math.sin(2 * theta_12) * (math.pow(math.sin(theta_23), 2) * math.pow(math.sin(theta_13), 2) - math.pow(math.cos(theta_23), 2)) - math.sin(2 * theta_23) * math.cos(2 * theta_12) * math.sin(theta_13)) + 0.5 * math.pow(math.sin(2 * theta_13), 2) * math.pow(math.sin(theta_23), 2)
            p_etau = 1 - p_ee - p_emu
            
            self.flavours.clear()
            self.add_neutrino_flavour(NeutrinoFlavour.ElectronNeutrino, [p_ee] * npoints)
            self.add_neutrino_flavour(NeutrinoFlavour.MuonNeutrino, [p_emu] * npoints)
            self.add_neutrino_flavour(NeutrinoFlavour.TauNeutrino, [p_etau] * npoints)


        elif self.oscillation_mode == "solar_matter_sun":
            # This part is specific and requires SolarMSWLookupTable
            # Initialise lookup table or load data from a file if it exists
            lookup_table_ee = DMCalcConstants.get_dmcalc_path() + "/DataBase/NeutrinoOscillations/lookup_table_solar_MSW_ee.txt"
            lookup_table_emu = DMCalcConstants.get_dmcalc_path() + "/DataBase/NeutrinoOscillations/lookup_table_solar_MSW_emu.txt"

            if not os.path.exists(lookup_table_ee) or not os.path.exists(lookup_table_emu):
                print("SolarMSWLookupTable generation...")
                lookup = SolarMSWLookupTable(0.01e-6, 1000e-6, 1, 1000)
                lookup.output()
                print("New SolarMSWLookupTable files have been created.")
            
            # Now using the lookup table
            #lookup = SolarMSWLookupTable(min_energy_GeV, max_energy_GeV, min_distance_km, max_distance_km) 
            #TODO ask Rob where these inputs are from / how do we calculate 
            #TODO ask Rob what lookup table is meant to look like

            lookup = SolarMSWLookupTable(1e-7, 1e3, 1, 1000000) 
            production_points = DMCalcConstants.get_dmcalc_path() + "/DataBase/NeutrinoFlux/production_points/" + self.name + ".txt"
            print(production_points)
            p_ee_vec = []
            p_emu_vec = []
            p_etau_vec = []

            # Handle the energy lookup (keV to GeV conversion)
            if len(self.pdf_data) == 1:
                energy = self.pdf_data[0][0] * 1e-6  # Convert to GeV #TODO check
                with open(production_points, "r") as file:
                        values = []
                        for line in file:
                            distance, fraction = map(float, line.split())
                            values.append((distance, fraction))
                fraction_sum = 0.0
                ee_prod_sum = 0.0
                emu_prod_sum = 0.0
                for distance, fraction in values:
                    print(distance)
                    fraction_sum += fraction
                    ee_prod_sum += lookup.get_value(energy, distance * DMCalcConstants.Rsun, 1) * fraction
                    emu_prod_sum += lookup.get_value(energy, distance * DMCalcConstants.Rsun, 2) * fraction
                p_ee_vec.append(ee_prod_sum / fraction_sum)
                p_emu_vec.append(emu_prod_sum / fraction_sum)
                p_etau_vec.append(1.0 - (ee_prod_sum / fraction_sum) - (emu_prod_sum / fraction_sum))
            else:
                for i in range(len(self.pdf_data) - 1):
                    energy_mid = 0.5e-3 * (self.pdf_data[i][0] + self.pdf_data[i + 1][0])
                    with open(production_points, "r") as file:
                        values = []
                        for line in file:
                            distance, fraction = map(float, line.split())
                            values.append((distance, fraction))
                    fraction_sum = 0.0
                    ee_prod_sum = 0.0
                    emu_prod_sum = 0.0
                    for distance, fraction in values:
                        fraction_sum += fraction
                        ee_prod_sum += lookup.get_value(energy_mid, distance * DMCalcConstants.Rsun, 1) * fraction
                        emu_prod_sum += lookup.get_value(energy_mid, distance * DMCalcConstants.Rsun, 2) * fraction
                    p_ee_vec.append(ee_prod_sum / fraction_sum)
                    p_emu_vec.append(emu_prod_sum / fraction_sum)
                    p_etau_vec.append(1.0 - (ee_prod_sum / fraction_sum) - (emu_prod_sum / fraction_sum))

            self.flavours.clear()
            self.add_neutrino_flavour(NeutrinoFlavour.ElectronNeutrino, p_ee_vec)
            self.add_neutrino_flavour(NeutrinoFlavour.MuonNeutrino, p_emu_vec)
            self.add_neutrino_flavour(NeutrinoFlavour.TauNeutrino, p_etau_vec)



    def pdf_data(self):
        data = []
        file = DMCalcConstants.get_dmcalc_path() + f"/DataBase/NeutrinoFlux/data/{self.name}.txt"
        with open(file, 'r') as f:
            for line in f:
                energy, flux = map(float, line.split())
                data.append([energy, flux])

        """
        #Check it plots reasonable spectrum
        data = np.array(data)
        plt.scatter(data[:, 0], data[:, 1], label="Flux Data")
        plt.xlabel('Energy')
        plt.ylabel('Flux')
        plt.legend()
        plt.savefig('example.png')
        plt.show()
        """
        return data
    
    def flavour_average(self, func: Callable[[float], float], flavour: str) -> float:
        """
        Averages a function over neutrino flux, weighted by the abundance of the specified flavour.

        :param func: Function to evaluate at each data point.
        :param flavour: Neutrino flavour ("e", "mu", "tau" or full names like "ElectronNeutrino").
        :return: Weighted average value.
        """
        # Map full neutrino names to their short forms

        full_to_short = {
            "ElectronNeutrino": "e",
            "MuonNeutrino": "mu",
            "TauNeutrino": "tau"
        }

        # Convert full name to short form if necessary
        if flavour in full_to_short:
            flavour = full_to_short[flavour]

        # Check if the flavour is valid
        if flavour not in self.flavours:
            total = 0
            #raise ValueError(f"Invalid flavour: {flavour}. Valid options are {list(self.flavours.keys())}")
        else:
            # Get the abundance for the specified flavour
            # Note flux units in MeV -- convert energy in keV to MeV
            abundance = self.flavours[flavour]
            # Handle line mode
            #TODO check with Rob which ones if any are meant to be line modes 
            if self.mode == FluxMode.Line:
                return abundance[0] * func(self.pdf_data[0][0]* 1e3)

            # Compute the total weighted average for non-line mode
            total = 0
            prev_val = func(self.pdf_data[0][0]* 1e3)
            

            for i in range(1, len(self.pdf_data)-1): 
                val = func(self.pdf_data[i][0])* 1e3 * abundance[i]
                #print(val)
                total += 0.5 * (val + prev_val) * abundance[i] * (self.pdf_data[i][0] - self.pdf_data[i - 1][0]) * 1e3
                #print(total)
                prev_val = val

        return total #in MeV units

    
    def get_total_flux(self, E_min: float = None, E_max: float = None) -> float:
        """
        Compute the total flux by integrating flux vs. energy.
        :return: Total flux (integrated over energy).
        """
        data = np.array(self.pdf_data)

        # Apply integration limits if specified
        if E_min is not None:
            data = data[data[:, 0] >= E_min] 
        if E_max is not None:
            data = data[data[:, 0] <= E_max]  

        # Check if there is only one data point, and handle that case
        if len(data) == 1:
            flux = data[:, 0] * 1e-3 * data[:, 1]*1e3  # Energy in keV to MeV
            flux = flux[0]
        else:
            # Assuming first column is energy (keV) and second column is flux (per s per cm^2 per MeV)
            flux = simps(x=data[:, 0] * 1e-3, y=data[:, 1]*1e3)  # Convert energy to MeV for integration

        # Return total flux in units of /s/cm^2
        return flux

