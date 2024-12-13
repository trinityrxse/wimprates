import math
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Tuple, Union
import numericalunits as nu
from lookup_solar_MSW_table import SolarMSWLookupTable
import os
from constants import *

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



class Nucleus:
    """
    Placeholder for Nucleus class.
    """
    def __init__(self, name: str, mass: float, charge: int):
        self.name = name
        self.mass = mass
        self.charge = charge

    def getA(self):
        return self.mass
    
    def getZ(self):
        return self.charge
    
    def getMGeV(self):
        return self.mass * nu.amu


class VNeutrinoCrossSection:
    """
    Abstract base class for neutrino cross-section calculations.
    """
    def dSigmadEr_cm2_keV(self, recoil_keV: float, neutrino_keV: float, nucleus: Nucleus) -> float:
        raise NotImplementedError

    def SetCouplings(self, neutrinoFlavour: NeutrinoFlavour):
        raise NotImplementedError

    def SetBinding(self, isRRPA: bool, Z_nuc: int):
        pass

class NeutrinoFlux:
    def __init__(self, name: str, scaling: float,
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
        self.add_neutrino_flavour(self.flavour_map[neutrino_flavour])

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

        if self.oscillation_mode == OscillationMode.NoneMode:
            self.flavours.clear()
            self.add_neutrino_flavour(NeutrinoFlavour.ElectronNeutrino, [1.0] * npoints)

        elif self.oscillation_mode == OscillationMode.VacuumSunProduction:
            p_ee = 1 - 0.5 * math.pow(math.sin(2 * theta_12), 2) * math.pow(math.cos(theta_13), 4) - 0.5 * math.pow(math.sin(2 * theta_13), 2)
            p_emu = -0.5 * math.sin(2 * theta_12) * math.pow(math.cos(theta_13), 2) * (math.sin(2 * theta_12) * (math.pow(math.sin(theta_23), 2) * math.pow(math.sin(theta_13), 2) - math.pow(math.cos(theta_23), 2)) - math.sin(2 * theta_23) * math.cos(2 * theta_12) * math.sin(theta_13)) + 0.5 * math.pow(math.sin(2 * theta_13), 2) * math.pow(math.sin(theta_23), 2)
            p_etau = 1 - p_ee - p_emu
            
            self.flavours.clear()
            self.add_neutrino_flavour(NeutrinoFlavour.ElectronNeutrino, [p_ee] * npoints)
            self.add_neutrino_flavour(NeutrinoFlavour.MuonNeutrino, [p_emu] * npoints)
            self.add_neutrino_flavour(NeutrinoFlavour.TauNeutrino, [p_etau] * npoints)

        elif self.oscillation_mode == OscillationMode.MatterSunProduction:
            # This part is specific and requires SolarMSWLookupTable
            # Initialize lookup table or load data from a file if it exists
            lookup_table_ee = DMCalcConstants.get_dmcalc_path() + "/DataBase/NeutrinoOscillations/lookup_table_solar_MSW_ee.txt"
            lookup_table_emu = DMCalcConstants.get_dmcalc_path() + "/DataBase/NeutrinoOscillations/lookup_table_solar_MSW_emu.txt"

            if not os.path.exists(lookup_table_ee) or not os.path.exists(lookup_table_emu):
                print("SolarMSWLookupTable generation...")
                lookup = SolarMSWLookupTable(DMCalcConstants.get_dmcalc_path())
                lookup.output()
                print("New SolarMSWLookupTable files have been created.")
            
            # Now using the lookup table
            lookup = SolarMSWLookupTable(DMCalcConstants.get_dmcalc_path())
            production_points = DMCalcConstants.get_dmcalc_path() + "/DataBase/NeutrinoFlux/production_points/" + self.name + ".txt"

            p_ee_vec = []
            p_emu_vec = []
            p_etau_vec = []

            # Handle the energy lookup (keV to GeV conversion)
            if len(self.pdf_data()) == 1:
                energy = self.pdf_data()[0][0] * 1e-3  # Convert to GeV
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
                    ee_prod_sum += lookup.get_value(energy, distance * DMCalcConstants.Rsun, 1) * fraction
                    emu_prod_sum += lookup.get_value(energy, distance * DMCalcConstants.Rsun, 2) * fraction
                p_ee_vec.append(ee_prod_sum / fraction_sum)
                p_emu_vec.append(emu_prod_sum / fraction_sum)
                p_etau_vec.append(1.0 - (ee_prod_sum / fraction_sum) - (emu_prod_sum / fraction_sum))
            else:
                for i in range(len(self.pdf_data()) - 1):
                    energy_mid = 0.5e-3 * (self.pdf_data()[i][0] + self.pdf_data()[i + 1][0])
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
        file = DMCalcConstants.GetDMCalcPath() + "/DataBase/NeutrinoFlux/Data/8B.txt"
        with open(file, 'r') as f:
            for line in f:
                energy, flux = map(float, line.split())
                data.append([energy, flux])
        return data
    
    def flavor_average(self, func: Callable[[float], float], flavour: str) -> float:
        if flavour not in self.flavours:
            return 0

        abundance = self.flavours[flavour]
        if self.mode == FluxMode.Line:
            return abundance[0] * func(self.pdf_data[0][0])

        total = 0
        prev_val = func(self.pdf_data[0][0])

        for i in range(1, len(self.pdf_data)):
            val = func(self.pdf_data[i][0]) * abundance[i]
            total += 0.5 * (val * self.pdf_data[i][1] + prev_val * self.pdf_data[i - 1][1]) * (self.pdf_data[i][0] - self.pdf_data[i - 1][0])
            prev_val = val

        return total


# Example function to be used with flavor_average
def example_function(x: float) -> float:
    return x * 0.1  # Simple scaling function for demonstration

# Main usage
def main():
    # Create a neutrino flux object
    flux = NeutrinoFlux(name="Neutrino_Solar", scaling=1.0, neutrino_flavour="e", oscillation_mode="solar_vac_sun")

    # Apply oscillations (this updates the flavours)
    flux.apply_oscillation()

    # Print the flavoured distribution after oscillations
    print("Neutrino Flavours After Oscillation:", flux.flavours)

    # Compute the flavour average for electron neutrino using a sample function
    avg_flux = flux.flavor_average(example_function, "e")
    print(f"Average flux for Electron Neutrino: {avg_flux}")

if __name__ == "__main__":
    main()