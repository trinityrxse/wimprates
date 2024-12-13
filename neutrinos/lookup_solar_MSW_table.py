import numpy as np
import os
from constants import *
from propagate import *

class SolarMSWLookupTable:
    def __init__(self, *args):
        if len(args) == 4:
            self._init_from_params(*args)
        elif len(args) == 12:
            self._init_with_propagator(*args)
        else:
            raise ValueError("Invalid number of arguments")

    def _init_with_propagator(self, density_file, deltaMsq21, deltaMsq32, theta12, theta13, theta23, delta, min_energy_GeV, max_energy_GeV, min_distance_km, max_distance_km, averaging):
        self.fMinEnergy_GeV = min_energy_GeV
        self.fMaxEnergy_GeV = max_energy_GeV
        self.fMinDistance_km = min_distance_km
        self.fMaxDistance_km = max_distance_km
        self.fSizey = 100 
        self.fSizex = 100

        self.fLookup_ee = np.zeros((self.fSizey, self.fSizex))
        self.fLookup_emu = np.zeros((self.fSizey, self.fSizex))

        delta_energy = (self.fMaxEnergy_GeV - self.fMinEnergy_GeV) / (self.fSizey - 1)
        delta_distance = (self.fMaxDistance_km - self.fMinDistance_km) / (self.fSizex - 1)

        propagator = Propagate(density_file, self.fMinEnergy_GeV, deltaMsq21, deltaMsq32, theta12, theta13, theta23, delta, int(averaging))

        for i in range(self.fSizey):
            energy = self.fMinEnergy_GeV + i * delta_energy
            propagator.change_params(energy, deltaMsq21, deltaMsq32, theta12, theta13, theta23, delta)
            for j in range(self.fSizex):
                distance = self.fMinDistance_km + j * delta_distance
                self.fLookup_ee[i, j] = propagator.get_transition_prob(distance, 1, 1, True)
                self.fLookup_emu[i, j] = propagator.get_transition_prob(distance, 1, 2, True)

    def _init_from_params(self, min_energy_GeV, max_energy_GeV, min_distance_km, max_distance_km):
        self.fMinEnergy_GeV = min_energy_GeV
        self.fMaxEnergy_GeV = max_energy_GeV
        self.fMinDistance_km = min_distance_km
        self.fMaxDistance_km = max_distance_km
        self.fSizey = 100  # Placeholder for array size
        self.fSizex = 100

        self.fEnergies_MeV = []
        self.fDistances_km = []
        delta_energy = (self.fMaxEnergy_GeV - self.fMinEnergy_GeV) / (self.fSizey - 1)
        delta_distance = (self.fMaxDistance_km - self.fMinDistance_km) / (self.fSizex - 1)

        for i in range(self.fSizey):
            energy = self.fMinEnergy_GeV + i * delta_energy
            distance = self.fMinDistance_km + i * delta_distance
            self.fEnergies_MeV.append(energy * 1000)  # GeV -> MeV
            self.fDistances_km.append(distance)

        self._load_lookup_tables()

    def _load_lookup_tables(self):
        base_path = os.path.join(DMCalcConstants.get_dmcalc_path(), "DataBase/NeutrinoOscillations")
        self.fLookup_ee = self._load_table(os.path.join(base_path, "lookup_table_solar_MSW_ee.txt"))
        self.fLookup_emu = self._load_table(os.path.join(base_path, "lookup_table_solar_MSW_emu.txt"))

    def _load_table(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error opening {path}")
        data = np.loadtxt(path, skiprows=6).reshape((self.fSizey, self.fSizex))
        return data

    def output(self):
        base_path = os.path.join(DMCalcConstants.get_dmcalc_path(), "DataBase/NeutrinoOscillations")
        self._save_table(os.path.join(base_path, "lookup_table_solar_MSW_ee.txt"), self.fLookup_ee)
        self._save_table(os.path.join(base_path, "lookup_table_solar_MSW_emu.txt"), self.fLookup_emu)

    def _save_table(self, path, table):
        with open(path, 'w') as f:
            f.write(f"{self.fSizey}\n")
            f.write(f"{self.fMinEnergy_GeV}\n")
            f.write(f"{self.fMaxEnergy_GeV}\n")
            f.write(f"{self.fSizex}\n")
            f.write(f"{self.fMinDistance_km}\n")
            f.write(f"{self.fMaxDistance_km}\n")
            np.savetxt(f, table.flatten())

    def binary_search(self, array, value):
        low, high = 0, len(array) - 1
        while low <= high:
            mid = (low + high) // 2
            if array[mid] == value:
                return mid
            elif array[mid] < value:
                low = mid + 1
            else:
                high = mid - 1
        return low - 1

    def get_value(self, energy_MeV, distance_km, transition):
        where_energy = self.binary_search(self.fEnergies_MeV, energy_MeV)
        where_distance = self.binary_search(self.fDistances_km, distance_km)

        if transition == 1:
            return self.fLookup_ee[where_energy, where_distance]
        elif transition == 2:
            return self.fLookup_emu[where_energy, where_distance]
        else:
            raise ValueError("Invalid transition index")
