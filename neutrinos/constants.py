import os

class DMCalcConstants:
    sin_sq_theta_12 = 0.307
    sin_sq_theta_13 = 0.0218
    sin_sq_theta_23 = 0.545
    sin_sq_theta_W = 0.231
    delta_msq_21 = 7.39e-5
    delta_msq_32 = 2.525e-3
    delta = 1.36
    min_energy_lookup_GeV = 0.001
    max_energy_lookup_GeV = 100
    min_distance_lookup_km = 0.001
    max_distance_lookup_km = 100000
    averaging_vacuum = 1
    Rsun = 696340  # Radius of the Sun in km
    Gf = 1.166e-17 #Fermi const in units keV^-2
    m_e_keV = 0.511
 

    @staticmethod
    def get_dmcalc_path():
        current_dir = os.getcwd()  # Get current working directory
        parent_dir = os.path.dirname(current_dir)  # Get the parent directory
        return parent_dir
    
ATOMIC_WEIGHT = dict(
    Xe=131.293,
    Ar=39.948,
    Ge=72.64,
    Si=28.0855
)