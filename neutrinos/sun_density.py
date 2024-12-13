import math
from typing import List

class DMCalcConstants:
    Rsun = 6.957e5  # Solar radius in kilometers
    n_avogadro = 6.02214076e23  # Avogadro's number

class SunDensity:
    """
    Class for loading and storing the electron density profile of the sun.
    """

    def __init__(self, density_file_name: str):
        """
        Initialise SunDensity with a given electron density profile file.
        """
        self.fDensityFileName = density_file_name
        self.fTraverseEdges_km = []
        self.fTraverseDistance_km = []
        self.fTraverseRhos_invcm3 = []
        self.fNumShells = 0

        self.set_density_profile()
        self.check_density_profile()

    def get_num_shells(self):
        return self.fNumShells
    
    def get_density_in_shell(self, index):
        """
        Get the number of shells within the solar electron density profile
        Returns number of shells within the solar electron density profile loaded in
        """
        return self.fTraverseRhos_invcm3[index]
    
    def get_distance_across_shell(self, index):
        return self.fTraverseDistance_km[index]

    def set_density_profile(self):
        """
        Reads an electron density profile from the file.
        Converts shell edges to km, densities to cm^-3.
        """
        try:
            with open(self.fDensityFileName, 'r') as data:
                temp_rhos = []
                for line in data:
                    r_dist, rho = map(float, line.split())
                    self.fTraverseEdges_km.append(r_dist * DMCalcConstants.Rsun)
                    temp_rhos.append(math.pow(10, rho) * DMCalcConstants.n_avogadro)  # cm^-3

            self.fNumShells = len(self.fTraverseEdges_km) - 1
            for i in range(self.fNumShells):
                self.fTraverseDistance_km.append(
                    self.fTraverseEdges_km[i + 1] - self.fTraverseEdges_km[i]
                )
                self.fTraverseRhos_invcm3.append(temp_rhos[i + 1])
        except FileNotFoundError:
            raise RuntimeError(f"Error opening file: {self.fDensityFileName}")
        except Exception as e:
            raise RuntimeError(f"Error processing density file: {e}")

    def check_density_profile(self):
        """
        Checks that the last density shell has sufficiently low electron density for vacuum averaging.
        """
        if self.fTraverseRhos_invcm3[-1] > 0.001 * DMCalcConstants.n_avogadro:
            raise RuntimeError("Last density shell has electron density greater than 0.001 * N_A")

    @staticmethod
    def binary_search(v: List[float], low: int, high: int, value: float) -> int:
        """
        Perform a binary search to find the greatest index in the vector `v` (between `low` and `high`)
        where the element is less than or equal to `value`.
        """
        if low > high:
            return -1

        if value >= v[high]:
            return high

        mid = (low + high) // 2

        if v[mid] == value:
            return mid

        if mid > 0 and v[mid - 1] <= value < v[mid]:
            return mid - 1

        if value < v[mid]:
            return SunDensity.binary_search(v, low, mid - 1, value)

        return SunDensity.binary_search(v, mid + 1, high, value)

    def get_shells_ahead(self, start_point_km: float) -> int:
        """
        Returns the number of density shells ahead of the given starting point in km.
        """
        edge = self.binary_search(self.fTraverseEdges_km, 0, self.fNumShells, start_point_km)
        if edge == -1:
            raise RuntimeError("start_point_km is less than all values in the density profile")

        return self.fNumShells - edge


  