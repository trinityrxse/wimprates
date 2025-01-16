from neutrino_rates import *
import unittest
import math
import test_neutrinos
import pandas as pd 
import pickle as pkl


class TestNeutrino(unittest.TestCase):

    def test_components_NR(self):
        plt.figure(figsize = (12,8))
        er = np.logspace(-3, 2, 1000)

       # with open('8Bdata.pkl', 'rb') as file:
       #     data = pkl.load(file)

        xe_target = Target("Xe", 131.29, 54)

        # Define interaction type (e.g., COHERENT)
        interaction_type = InteractionType.COHERENT

        # Required flux component (e.g., "8B" neutrinos from the sun)
        required_fluxes = "All", "DSN", "Atmospheric", "8B", "HEP", "PP", "PEP", "CNO", "7Be"

        # Initialise the neutrino rate calculation for xe target
        nurate = NeutrinoRate(required_fluxes, interaction_type, xe_target)


        fig1, ax1 = plt.subplots(figsize=(10, 6))  
        fig2, ax2 = plt.subplots(figsize=(10, 6))  

        overall_flux = 0  # Initialize overall flux
        for name in required_fluxes:
            nurate.set_required_fluxes(name)
            total = 0
            rates = []
            rates_per_keV = []

            for e in er:
                rate = nurate.get_rate(e)
                rate_per_keV = rate / e
                rates.append(rate)
                rates_per_keV.append(rate_per_keV)


            if name != 'All':
                total += nurate.f_flux.get_total_flux_cm2s()
                ax1.loglog(er, rates, label=name)
                ax2.loglog(er, rates_per_keV, label=name)
                print(f'Total Flux: {total} for {name}')
                overall_flux += total

        #ax1.loglog(data['Recoil Energy [keV]'], data['Rate [Events / ton/year / keV]'], label='pkl file')
        #dataprime = data['Rate [Events / ton/year / keV]'] / rate
        #print(dataprime)
        #ax1.scatter(data['Recoil Energy [keV]'], dataprime, label='ratio')
        ax1.set_xlabel('Recoil Energy [keV]', fontsize=16)
        ax1.set_ylabel('Rate [Events / ton / year]', fontsize=16)
        ax1.set_title('Neutrino Recoil Rates (NR)', fontsize=20)
        ax1.grid(which='both', linestyle='--')
        ax1.legend(fontsize=12)
        fig1.tight_layout()
        fig1.savefig("outputs/NR_spectrum_rates.png")

        ax2.set_xlabel('Recoil Energy [keV]', fontsize=16)
        ax2.set_ylabel('Rate [Events / ton / year / keV]', fontsize=16)
        ax2.set_title('Neutrino Recoil Rates per keV (NR)', fontsize=20)
        ax2.grid(which='both', linestyle='--')
        ax2.legend(fontsize=12)
        fig2.tight_layout()
        fig2.savefig("outputs/NR_spectrum_rates_per_keV.png")

        plt.show()

        print(f"Overall Flux: {overall_flux}, All Flux: {nurate.spectrum().get_total_flux_cm2s()}")

    def test_flux(self):

         # Required flux component (e.g., "8B" neutrinos from the sun)
        required_fluxes = "All", "DSN", "Atmospheric", "8B", "HEP", "PP", "PEP", "CNO", "7Be"

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

        c = {"dsnbflux_8": 'tab:cyan', 
             "dsnbflux_5": 'tab:blue', 
             "dsnbflux_3": 'tab:orange',
             "AtmNu_e": 'tab:green', 
             "AtmNu_ebar": 'tab:red',
             "AtmNu_mu": 'tab:purple',
             "AtmNu_mubar": 'tab:brown',
             "8B": 'tab:brown',
             "hep": 'tab:green',
             "pep": 'tab:orange',
             "pp": 'tab:blue',
             "13N": 'tab:pink',
             "15O": 'tab:gray',
             "17F": 'tab:olive',
             "7Be_384.3keV": 'tab:red',
             "7Be_861.3keV": 'tab:purple'
             }
        
        for name in required_fluxes:
            if name == 'All':
                pass
            else:
                for key in flux_map[name]:
                    if key in ["dsnbflux_8", "dsnbflux_5", "dsnbflux_3", "AtmNu_e", "AtmNu_ebar", "AtmNu_mu", "AtmNu_mubar"
                               ]:
                        flux = NeutrinoFlux(name=key, scaling=1.0, neutrino_flavour="e", oscillation_mode="none")
                        print('skip osc')
                    else:
                        flux = NeutrinoFlux(name=key, scaling=1.0, neutrino_flavour="e", oscillation_mode="solar_vac_sun")
                        flux.apply_oscillation()

                    flux_pdf = np.array(flux.pdf_data)
   

                    energy, amp = flux_pdf[:,0], flux_pdf[:,1] #TODO these need to be scaled by arbitrary powers of 10 to match Rob's
                                            #WHY?? I am just plotting what should be the exact same numbers as Rob


                    if len(energy) == 1:
                        plt.plot([energy[0], energy[0]], [0, 1e5], color = c[key], ls='--', label=key)
                    
                    else:
                        # Find index of the minimum energy
                        min_energy_idx = np.argmin(energy)
                        min_energy = energy[min_energy_idx]
                        corresponding_amp = amp[min_energy_idx]

                        if min_energy > 1e-3:
                            # Plot vertical line at the minimum energy
                            plt.plot([min_energy, min_energy], [0, corresponding_amp], color = c[key])
                            plt.loglog(energy, amp, label=flux.name, color=c[key])

                        else:
                            plt.loglog(energy, amp, label=flux.name, color=c[key])

        #plt.xlim(1e-3, 10000)
        #plt.ylim(1e-7, 1e4)
        plt.xlabel('Neutrino Energy [MeV]', fontsize = 14)
        plt.legend(fontsize = 8, loc = 'upper right')
        plt.ylabel('Flux [1/cm$^2$/s/MeV]', fontsize = 14)
        plt.title('Neutrino Flux', fontsize = 20)
        plt.savefig('outputs/NuFluxes.png')
        plt.show()
        print('complete')

    def test_ER_methods(self):
        xe_target = Target("Xe", 131.29, 54)

        interaction_types = [InteractionType.EW_FREE_ELECTRON, 
                             InteractionType.EW_STEPPING, 
                             InteractionType.EW_RRPA
                             ]

        # Required flux component (e.g., "8B" neutrinos from the sun)
        required_fluxes = "All", "DSN", "Atmospheric", "8B", "HEP", "PP", "PEP", "CNO", "7Be"

        fig1, ax1 = plt.subplots(figsize=(10, 6))  
        fig2, ax2 = plt.subplots(figsize=(10, 6)) 

        datasets = []
        er = np.logspace(-3, 2, 100)
        for int_type in interaction_types:
            nurate = NeutrinoRate(required_fluxes, int_type, xe_target)
            for name in required_fluxes:
                nurate.set_required_fluxes(name)
                rates = []
                rates_per_keV = []

                for e in er:
                    rate = nurate.get_rate(e)
                    rates.append(rate)
                    rates_per_keV.append(rate/e)

                print(max(rates), 'max rate')
                data = pd.DataFrame({'Energy': er, 'Rate': rates, 'Rate per keV': rates_per_keV})
                datasets.append(data)
            
            concatenated_df = pd.concat(datasets, ignore_index=True)
                
            bin_edges = np.logspace(-3, 2, 100)
            bin_labels = [f"{bin_edges[i]}-{bin_edges[i + 1]}" for i in range(len(bin_edges) - 1)]
            concatenated_df['EnergyBin'] = pd.cut(concatenated_df['Energy'], bins=bin_edges, labels=bin_labels, right=False)
     
            binned_df = concatenated_df.groupby('EnergyBin', as_index=False).agg({
                'Rate': 'sum', 'Rate per keV': 'mean'  
            })
            print(binned_df)

            ax1.loglog(binned_df["EnergyBin"], binned_df["Rate"], label = int_type)
            ax2.loglog(binned_df["EnergyBin"], binned_df["Rate per keV"], label = int_type)


        #plt.xlim(1e-3, 10000)
        #plt.ylim(1e-7, 1e4)
        ax1.set_xscale("log")
        ax1.set_xlabel('Recoil Energy [keV]', fontsize = 14)
        ax1.legend(fontsize = 8, loc = 'lower right')
        ax1.set_ylabel('Rate [Events/ton/year/keV]', fontsize = 14)
        ax1.set_title('Neutrino Recoil Rates (ER)', fontsize = 20)
        ax1.grid(which='both', linestyle='--')
        fig1.tight_layout()
        fig1.savefig('outputs/ER_comparison.png')

        ax2.set_xscale("log")
        ax2.set_xlabel('Recoil Energy [keV]', fontsize=16)
        ax2.set_ylabel('Rate [Events / ton / year / keV]', fontsize=16)
        ax2.set_title('Neutrino Recoil Rates per keV (ER)', fontsize=20)
        ax2.grid(which='both', linestyle='--')
        ax2.legend(fontsize=12)
        fig2.tight_layout()
        fig2.savefig("outputs/ER_comparison_per_keV.png")

        xe_target = Target("Xe", 131.29, 54)

        # Define interaction type (e.g., COHERENT)
        interaction_types = [InteractionType.EW_FREE_ELECTRON, InteractionType.EW_STEPPING, InteractionType.EW_RRPA]

        # Required flux component (e.g., "8B" neutrinos from the sun)
        required_fluxes = "All", "DSN", "Atmospheric", "8B", "HEP", "PP", "PEP", "CNO", "7Be", "7Be_PP_CNO"

        er = np.logspace(-1, 2, 50)
        for int_type in interaction_types:
            nurate = NeutrinoRate(required_fluxes, int_type, xe_target)
            nurate.set_required_fluxes(required_fluxes)
            total = 0
            rates = []
            rates_per_keV = []
            for e in er:
                rate = nurate.get_rate(e)
                rates.append(rate)
                rates_per_keV.append(rate/e)


            total += nurate.f_flux.get_total_flux_cm2s()
            plt.loglog(er, rates, label = int_type)

            print('Total Flux: ', total, ' for ', int_type)
        
        plt.xlim(1e-3, 10000)
        plt.ylim(1e-7, 1e4)
        plt.xlabel('Recoil Energy [keV]', fontsize = 14)
        plt.legend(fontsize = 8, loc = 'upper right')
        plt.ylabel('Rate [Events/ton/year/keV]', fontsize = 14)
        plt.title('Neutrino Recoil Rates (ER)', fontsize = 20)
        plt.savefig('outputs/ER_comparison.png')
        plt.show()

    def test_components_ER(self):
        xe_target = Target("Xe", 131.29, 54)

        interaction_type = InteractionType.EW_RRPA

        # Required flux component (e.g., "8B" neutrinos from the sun)
        required_fluxes = "All", "DSN", "Atmospheric", "8B", "HEP", "PP", "PEP", "CNO", "7Be"

        # Initialise the neutrino rate calculation for xe target
        nurate = NeutrinoRate(required_fluxes, interaction_type, xe_target)

        fig1, ax1 = plt.subplots(figsize=(10, 6))  
        fig2, ax2 = plt.subplots(figsize=(10, 6))  

        overall_flux = 0  # Initialize overall flux

        er = np.logspace(-1, 5, 1000)
        for name in required_fluxes:
            nurate.set_required_fluxes(name)
            total = 0
            rates = []
            rates_per_keV = []

            for e in er:
                rate = nurate.get_rate(e)
                rate_per_keV = rate / e
                rates.append(rate)
                rates_per_keV.append(rate_per_keV)


            if name != 'All':
                total += nurate.f_flux.get_total_flux_cm2s()
                ax1.loglog(er, rates, label=name, alpha=0.5)
                ax2.loglog(er, rates_per_keV, label=name, alpha=0.5)
                print(f'Total Flux: {total} for {name}')
                overall_flux += total

        ax1.set_xlabel('Recoil Energy [keV]', fontsize=16)
        ax1.set_ylabel('Rate [Events / ton / year]', fontsize=16)
        ax1.set_title('Neutrino Recoil Rates (ER)', fontsize=20)
        ax1.grid(which='both', linestyle='--')
        ax1.legend(fontsize=12)
        fig1.tight_layout()
        fig1.savefig("outputs/ER_spectrum_rates.png")

        ax2.set_xlabel('Recoil Energy [keV]', fontsize=16)
        ax2.set_ylabel('Rate [Events / ton / year / keV]', fontsize=16)
        ax2.set_title('Neutrino Recoil Rates per keV (ER)', fontsize=20)
        ax2.grid(which='both', linestyle='--')
        ax2.legend(fontsize=12)
        fig2.tight_layout()
        fig2.savefig("outputs/ER_spectrum_rates_per_keV.png")

        plt.show()

        print(f"Overall Flux: {overall_flux}, All Flux: {nurate.spectrum().get_total_flux_cm2s()}")

"""    
    def test_components_NR(self):
        plt.figure(figsize = (12,8))
        er = np.logspace(-4, 2, 1000)

        xe_target = Target("Xe", 131.29, 54)

        # Define interaction type (e.g., COHERENT)
        interaction_type = InteractionType.COHERENT

        # Required flux component (e.g., "8B" neutrinos from the sun)
        required_fluxes = "All", "DSN", "Atmospheric", "8B", "HEP", "PP", "PEP", "CNO", "7Be", "7Be_PP_CNO"

        # Initialise the neutrino rate calculation for xe target
        nurate = NeutrinoRate(required_fluxes, interaction_type, xe_target)


        fig1, ax1 = plt.subplots(figsize=(10, 6))  
        fig2, ax2 = plt.subplots(figsize=(10, 6))  

        overall_flux = 0  

        for name in required_fluxes:
            nurate.set_required_fluxes(name)
            total = 0
            rates = []
            rates_per_keV = []

            for e in er:
                rate = nurate.get_rate(e)
                rate_per_keV = rate / e
                rates.append(rate)
                rates_per_keV.append(rate_per_keV)


            if name != 'All':
                total += nurate.f_flux.get_total_flux_cm2s()
                ax1.loglog(er, rates, label=name)
                ax2.loglog(er, rates_per_keV, label=name)
                print(f'Total Flux: {total} for {name}')
                overall_flux += total

        ax1.set_xlabel('Recoil Energy [keV]', fontsize=16)
        ax1.set_ylabel('Rate [Events / ton / year]', fontsize=16)
        ax1.set_title('Neutrino Recoil Rates (NR)', fontsize=20)
        ax1.grid(which='both', linestyle='--')
        ax1.legend(fontsize=12)
        fig1.tight_layout()
        fig1.savefig("outputs/NR_spectrum_rates.png")

        ax2.set_xlabel('Recoil Energy [keV]', fontsize=16)
        ax2.set_ylabel('Rate [Events / ton / year / keV]', fontsize=16)
        ax2.set_title('Neutrino Recoil Rates per keV (NR)', fontsize=20)
        ax2.grid(which='both', linestyle='--')
        ax2.legend(fontsize=12)
        fig2.tight_layout()
        fig2.savefig("outputs/NR_spectrum_rates_per_keV.png")

        plt.show()

        print(f"Overall Flux: {overall_flux}, All Flux: {nurate.spectrum().get_total_flux_cm2s()}")
"""
 