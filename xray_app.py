import streamlit as st
import numpy as np # Numpy for handling arrays
import matplotlib.pyplot as plt # Matplotlib to do the plotting
import tabulate # Tables
from numba import jit #just in time compilation - faster code
import simulation_functions as sim
import generate_plots as gen

plt.rcParams.update({'axes.grid': True, 'grid.linestyle' : "--", 'figure.figsize' : (9.5,6)})

def main():
    '''
    Function for running the Streamlit application
    User can interact with simulation and plots
    '''

    # Defining some session state variables
    # These are used to avoid having to rerun simulations extra times

    # Boolean variables to check if simulation has been run
    if 'simulation_1' not in st.session_state:
        st.session_state['simulation_1'] = False
    
    if 'simulation_2' not in st.session_state:
        st.session_state['simulation_2'] = False

    if 'simulation_3' not in st.session_state:
        st.session_state['simulation_3'] = False

    # Variables to store the figures
    if 'fig1' not in st.session_state:
        st.session_state['fig1'] = None
    
    if 'fig2' not in st.session_state:
        st.session_state['fig2'] = None

    # fig2 is a plot showing attenuation coefficient and does not require any simulation
    # Therefore it is not stored as session_state variable

    if 'fig4' not in st.session_state:
        st.session_state['fig4'] = None

    if 'fig5' not in st.session_state:
        st.session_state['fig5'] = None

    if 'fig6' not in st.session_state:
        st.session_state['fig6'] = None

    if 'fig7' not in st.session_state:
        st.session_state['fig7'] = None
    
    if 'fig8' not in st.session_state:
        st.session_state['fig8'] = None
    
    if 'fig9' not in st.session_state:
        st.session_state['fig9'] = None
    
    if 'fig10' not in st.session_state:
        st.session_state['fig10'] = None

    # Main title of application
    st.title("Monte Carlo Simulation of X-ray imaging.")

    ############################################
    ######//////    First part      \\\\\\######
    ############################################

    st.subheader("Stability of Monte Carlo Simulation")
    st.write("In this part a one-dimensional beam of photons will be sent through a material with dampening coefficient, $\mu$. \
              The result from the Monte-Carlo method and the analytical solution will be compared. \
              In the end the stability of the Monte-Carlo method will be looked into by comparing different number of photons and different steplengths $\Delta x$.\
              This part is mainly to check that the numerical solver corresponds with the analytical solution.")
    
    with st.expander("Specify parameters for simulation"):
        # Make it possible for user to change default paramters for first part
        width_1 = st.number_input("Width (cm)", 1, 100, 10, 1)
        n_steps_1 = st.number_input("Number of steps", 1, 10000, 100, 1)
        x_1 = np.linspace(0, width_1, n_steps_1 + 1)
        n_photons_1 = st.number_input("Number of photons", 1, 10000, 10000, 1)
        my1 = 0.1 * np.ones(n_steps_1)

    # Have option for user to run the simulation
    if st.button("Run simulation"):
        fig1, fig2 = gen.simulation_1(width_1, n_steps_1, x_1, n_photons_1, my1)
        st.session_state['fig1'] = fig1
        st.session_state['fig2'] = fig2

        # Update to say that first simulation has been run
        st.session_state['simulation_1'] = True
    
    if st.session_state['simulation_1']:
        st.pyplot(st.session_state['fig1'])
        st.write("The plot above shows the accuracy of the numerical simulation for different number of photons.\
                 From the plot it is clear to see that as the number of photons in the photon beam increases, the Monte Carlo approximation approaches the analytical solution. ")
        
        st.pyplot(st.session_state['fig2'])
        st.write("The plot shows the numerical approximation compared to the analytical one for different steplenghts.\
                 From the plot it is apparent that a large number of steps (a small steplength) is needed to capture the curvature of the analytical solution. ")


    #############################################
    ######//////    Second part      \\\\\\######
    #############################################

    st.subheader("Contrast between bone and tissue")
    st.write("In this part two different photon beams will be simulated. One will go through only tissue, and the other will first go through tissue, \
             then through bone and in the end through tissue again. All three layers have the same thickness. Their contrast will be compared at different energy levels. \
             The attenuation of bone and tissue at different energies is given in the data folder. \
             The data file for bone was collected from \
             https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/bone.html, \n\
             and the data file for (soft) tissue was collected from \
             https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/tissue.html.\n\
             The first column contains the energy, and the second one the mass-attenuation coefficient. To get back the attentuation coefficient \
             $\mu $, these values will have to be devided by the density, $\rho$. I will use $\rho_{tissue} = 1.02 g/cm^3$ and $\rho_{bone} = 1.92 g/cm^3$.\n\
             This part of the project is supposed to simulate x-rays, where some of the rays will move through some soft tissue and some bone, \
             and others will move through only tissue. After the simulation is finished, the difference in concentration will be compared to see if any contrast may be visible.\
             Since the energy of x-rays used in medicine lies between 10keV and 100keV, the simulations in this part will consider energies in this range.")
    
    with st.expander("Specify parameters for simulation"):
        # Make it possible for user to change default paramters for second part
        width_2 = st.number_input("Width (cm)", 1, 100, 3, 1, key = 21)
        n_steps_2 = st.number_input("Number of steps", 1, 10000, 1000, 1, key = 22)
        n_photons_2 = st.number_input("Number of photons", 1, 10000, 10000, 1, key = 23)

    tissue_dens = 1.02 #g/cm^3
    bone_dens = 1.92 #g/cm^3
    # reading in the datafiles for attenuation coefficients
    energy_tissue, mu_tissue = np.loadtxt("data/tissue.txt", delimiter=',', unpack=True)
    energy_bone, mu_bone = np.loadtxt("data/bone.txt", delimiter=',', unpack=True)
    # Multiplying with density to get correct attenuation coefficient
    mu_tissue *= tissue_dens
    mu_bone *= bone_dens
    energy_lower, energy_higher = 1e-2, 1e-1

    if st.checkbox("Display attenuation coefficient", True):
        fig3 = plt.figure()
        plt.loglog(energy_tissue, mu_tissue, label = "Tissue")
        plt.loglog(energy_bone, mu_bone, label = "Bone")
        plt.title("Attentuation coefficient as function of density")
        plt.xlabel("Energy [MeV]"); plt.ylabel("$\mu$ $[cm^{-1}]$"); plt.legend(); plt.ylim(1e-2, 1e4)
        plt.vlines(energy_lower, 0, 1e4, color = "grey"); plt.vlines(energy_higher, 0, 1e4, color = "grey")
        st.pyplot(fig3)
        st.write("The plot shows the attenuation coefficient of bone and tissue at varying energies.\
                  It can be noted than the attenuation coefficient of bone always stays higher than tissue.")
    
    if st.button("Run simulation", key = 2):
        fig4, fig5, fig6, fig7, fig8 = gen.simulation_2(width_2, n_steps_2, n_photons_2)
        st.session_state['fig4'] = fig4
        st.session_state['fig5'] = fig5
        st.session_state['fig6'] = fig6
        st.session_state['fig7'] = fig7
        st.session_state['fig8'] = fig8
        st.session_state['simulation_2'] = True

    if st.session_state['simulation_2']:
        st.pyplot(st.session_state['fig4'])
        st.write("The plot shows the intensity of a photon beam going through bone and tissue. Since the attenuation coefficient of bone is higher than tissue\
                it is expected that the intensity at the detector is lower. This is observed in the plot above. Because the of the varying attenuation\
                coefficient, the difference in intensity also varies. The difference is biggest around 30-40 keV. However, instead of considering the difference directly,\
                it might be more usefull to look at the contrast.")
        
        st.pyplot(st.session_state['fig5'])
        st.write("The plot displays the contrast between a beam going through bone and one going throug tissue. The contrast looks to be the biggest for lower energy photon beams\
                 (but high enough energy for the photons to get through). It seems reasonable that the contrast starts at 1, since at 15keV some photons are able to move through the tissue, but not the bone.")
                 
        st.write("It might also be interesting to get an idea of how many photons are required for the detector to be able to detect the beam after it has passed through. \
                 If we assume that an intensity of 10MeV is required for the detector to be able to detect a signal, we can use the previously calculated results to find the necessary number of photons.\n\
                 Since the intensity of the tissue beam for the first energy and the intensity of the bone beam for the two first energies are numerically zero, they cannot be used to find the correct number of photons.")

        st.pyplot(st.session_state['fig6'])
        st.write("The plot shows the required number of photons to reach 10MeV at the detector. As expected,\
                  the plots show that a higher number of photons are required for the beam travelling through bone. \
                  Furthermore, as the energy increases and the percentage of photons passing through increases, \
                  the slopes flatten out and the distance to the minimal number of photons decreases.\n\
                  As mentioned before the overall goal is to maximize the contrast, while also minimizing the quantity of absorbed photons. \
                  For simplicity we will for the moment ignore Compton-scattering and assume that all of the photons\
                  that doesn't make it through the material are absorbed. Even though this is a rough approximation,\
                  it can give a good idea of which energy region that gives the best trade-off between a good contrast and a small absorbed dosage.\
                  The absorbed dosage is given by equation (4). For the beam that passes through 1/3 tissue, 1/3 bone and 1/3 tissue,\
                  we must know how many photons that were absorbed in each of the three sections.")

        st.pyplot(st.session_state['fig7'])
        st.write("The plot shows that the absorbed dosage decreases as the increases. This might seem counterintuitive, but can be explained by two effects.\
                  The first reason is that the attenuation coefficient decreases as the energy increases, allowing more photons to pass through.\
                  The second reason comes from the fact that the number of photons is set so that the energy at the detector of the beam moving through tissue is constant.\
                  In other words, the initial energy of the beam varies.")
        
        st.pyplot(st.session_state['fig8'])
        st.write("The figure shows the absorbed dosage and the contrast for energy levels ranging from 20keV to 100keV.\
                  Since a high contrast and a low absorbed quantity is to be desired, it is difficult to know how to choose\
                  the best combination. The absorbed dosage decreases quickly untill an energy level of 30keV and the\
                  contrast is quite high untill energy levels of 40-50 keV. Therefore it might be that the optimal energy\
                  level is somewhere between 30keV and 40keV.")



    ############################################
    ######//////    Third part      \\\\\\######
    ############################################

    st.subheader("X-ray imaging of 3D objects")
    st.write("In this part photon beams will be sent through different 3d objects at different energies instead of the 2d objects from before.\
              2d images will be created by sending the photon beams through the three different axes.\
              The matrices describing the attenuation coefficient of two unspecified objects at different grid points can be found in the data folder.\
              The 3d objects have one matrix for different energy levels. From these matrices the goal is to figure out which object is described\
              by sending photons through the material from different directions. To this end, the most interesting quantity to measure is how many\
              photons reach the detector after passing through the material.")
    
    with st.expander("Specify parameters for simulation"):
        n_photons_3 = st.number_input("Number of photons", 1, 10000, 1000, 1, key = 3)

    # Reading in the 3d-objects
    objects1 = np.array([np.load("data/object1_20keV.npy"), 
                        np.load("data/object1_50keV.npy"), 
                        np.load("data/object1_100keV.npy")])

    objects2 = np.array([np.load("data/object2_25keV.npy"), 
                        np.load("data/object2_50keV.npy"), 
                        np.load("data/object2_75keV.npy")])

    # Energies of photon beams
    energies3_1 = np.array([20, 50, 100])
    energies3_2 = np.array([25, 50, 75])

    # Widths of objects in x, y, z direction (cm).
    widths3_1 = np.array([6.5, 44.6, 44.6])
    widhts3_2 = np.array([12, 12, 10])

    if st.button("Run simulation", key = 31):
        fig9, fig10 = gen.simulation_3(n_photons_3, objects1, objects2, widths3_1, widhts3_2)
        st.session_state['fig9'] = fig9
        st.session_state['fig10'] = fig10
        st.session_state['simulation_3'] = True
    
    if st.session_state['simulation_3']:
        st.pyplot(st.session_state['fig9'])
        st.pyplot(st.session_state['fig10'])
        st.write("The first object seems to to display some organic matter. It is difficult to say for sure,\
                  but it could be a section of the thorax. The photon beams going through the yz-plane gives\
                  the most information about the object. The attenuation coefficient varies with energy, which\
                  is the reason the plots for the different energies look different. \n\
                  Lower energies make the wavy parts more visible, whereas higher energies gives higher contrast\
                  between sections in the interior. Depending of what is of interest, different energy levels may be appropriate.\
                  The images of the first object look quite similar to how traditional x-ray images might look.\n\
                  The second object seems to be a coffee mug. The mug is visible for all energy levels, but a higher energy level\
                  gives a higher contrast between the edge of the mug and the interior of the mug. The brighter line at the top of\
                  the mug might mean that it is filled with coffe up to that point.")

if __name__ == "__main__":
    main()