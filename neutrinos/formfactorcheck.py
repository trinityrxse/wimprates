import numpy as np
from scipy.special import spherical_jn

# Constants
INV_GEV_TO_FM = 5.067731  # Conversion factor [GeV]^-1 -> [fm]
PI = np.pi

# Spherical Bessel function j1(qR)
def sph_bessel_j1(x):
    if x == 0:
        return 1.0  # Avoid division by zero for small x
    return (np.sin(x) - x * np.cos(x)) / (x**2)

# Calculate momentum of a recoiling particle in [fm^-1]
def recoil_to_q_ifm(Er_keV, mass_GeV):
    #print(np.sqrt((2 * mass_GeV * Er_keV + Er_keV**2 / 1e6) / 1e6) / INV_GEV_TO_FM, 'q')
    return np.sqrt((2 * mass_GeV * Er_keV + Er_keV**2 / 1e6) / 1e6) / INV_GEV_TO_FM

# Translate momentum to recoil energy in keV
def q_to_recoil(q, mass_GeV):
    return (q * INV_GEV_TO_FM)**2 / (2 * mass_GeV)

# Helm form factor (with Gaussian smearing)
def helm(q, R):
    sparam = 0.9  # [fm]
    qR = q * R
    #print(qR, 'qR')
    if qR < 1e-12:
        return 1.0
    #print((3 * sph_bessel_j1(qR)) * ((np.exp(-0.5 * (q * sparam)**2))/qR))
    return (3 * sph_bessel_j1(qR)) * ((np.exp(-0.5 * (q * sparam)**2))/qR)

# Helm radius in [fm]
def helm_radius_fm(A):
    cparam = 1.23 * (A**(1.0 / 3.0)) - 0.6  # [fm]
    aparam = 0.52  # [fm]
    sparam = 0.9   # [fm]
    term = (7.0 / 3.0) * (PI**2) * (aparam**2) - 5 * (sparam**2)
    return np.sqrt(cparam**2 + term)

# Helm form factor for a nucleus
def helm_form_factor(Er_keV, A, mass_GeV):
    R_eff = helm_radius_fm(A)  # [fm]
    q = recoil_to_q_ifm(Er_keV, mass_GeV)  # [fm^-1]
    return helm(q, R_eff)

# Top-hat form factor
def tophat(q, R):
    x = q * R
    if x < 1e-12:
        return 1.0
    return 3 * sph_bessel_j1(x) / x

# Top-hat form factor for a nucleus
def tophat_form_factor(Er_keV, A, mass_GeV):
    cparam = 1.23 * (A**(1/3)) - 0.6  # [fm]
    aparam = 0.52  # [fm]
    sparam = 0.9   # [fm]
    R = np.sqrt(cparam**2 + (7.0 / 3.0) * (PI**2) * (aparam**2) - 5 * (sparam**2))  # [fm]
    q = recoil_to_q_ifm(Er_keV, mass_GeV)  # [fm^-1]
    return tophat(q, R)
