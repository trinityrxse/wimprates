# Neutrino Data
Find below some relevant information regarding the origin of the data in this repository.

## Solar neutrinos
The shape of the spectra of all the neutrino sources in `data/` comes from [Billard et al](https://arxiv.org/abs/1307.5458). The normalization factors in `TotalFluxes.txt` are based on the discussion in [Conventions for reporting results from direct dark matter searches](https://drive.google.com/file/d/1VoUzgnKjnnRQq0z89LMIc8rJ99MGb30w/view?usp=sharing).

## Atmospheric neutrinos
The best predictions on the atmospheric neutrino flux in the sub-GeV regime, the most relevant part of the atmospheric spectrum for dark matter searches, are currently based on the [FLUKA simulations](https://www.sciencedirect.com/science/article/abs/pii/S0927650505000526?via%3Dihub). The combined flux is equal to `10.5 \pm 2.1 nu/cm2/s`.

Note that the data files were updated in February 2021. These files were originally obtained by digitizing the relevant plot in the Billard et al paper, but this carries an intrinsic error due to the precision in the clicking abilities of the user. The new files were retrived from [this code repository](https://zenodo.org/record/3653516), and correspond to data coming directly from the FLUKA paper.

## Diffuse supernova neutrinos
The diffuse supernova neutrino background (DSNB) comes from the cumulative flux of neutrinos from supernova explosions over the hisotry of the universe. The neutrino spectrum of a core-collapse supernova is well approximated by a Fermi-Dirac (FD) distribution, with temperatures ranging in the 3 to 8 MeV. Based on [this reference](https://iopscience.iop.org/article/10.1086/375130), a 3 and a 5 MeV FD distribution are assigned to the electron and anti-electron neutrino components, while an 8 MeV FD distribution is assigned to the total contribution from all the rest of components. The total flux is equal to `86 \pm 43 nu/cm2/s`. The large uncertainty is due to large theoretical uncertainties in this calculation, see [Beacom 2010](https://www.annualreviews.org/doi/10.1146/annurev.nucl.010909.083331) for more information.

Both the shape and normalization of the spectra were digitized from the Billard et al paper, which are based on the references above.
