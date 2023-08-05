import streamlit as st
import numpy as np # Numpy for handling arrays
import matplotlib.pyplot as plt # Matplotlib to do the plotting
import tabulate # Tables
from numba import jit #just in time compilation - faster code
import simulation_functions as sim


def main():
    '''
    Function for running the Streamlit application
    User can interact with simulation and plots
    '''

    # Main title of application
    st.title("Monte Carlo Simulation of X-ray imaging.")

    # Subtitle of first part
    st.subheader("Stability of Monte Carlo Simulation")
    with st.expander("Choose parameters for first part"):
        # Make it possible for user to change default paramters for first part
        st.markdown("### Specify paramters for first part")
        width_1 = st.number_input("Width (cm)", 1, 100, 10, 1)
        n_steps_1 = st.number_input("Number of steps", 1, 10000, 100, 1)
        x_1 = np.linspace(0, width_1, n_steps_1 + 1)
        n_photons_1 = st.number_input("Number of photons", 1, 10000, 10000, 1)
        my1 = 0.1 * np.ones(n_steps_1)

    # Have option for user to run the simulation
    if st.button("Run simulation"):

        # Analytical solution given the parameters:
        I_analytical = np.exp(-my1[0] * x_1) # For this part we assume attenuation coefficient to be constant

        # Figure 1 showing the numerical approximation for different choices of number of photons
        fig1 = plt.figure(figsize = (9,7))
        plt.plot(x_1, I_analytical, label = 'Analytical', linestyle = '--', color = 'black', linewidth = 1)
        number_of_photons = [10, 50, 100, 1000, 10000, 100000]

        for i, n_photon in enumerate(number_of_photons):
            # Going through the different numbers of photons
            I_numerical = sim.simulate_photons(n_photon, n_steps_1, width_1, my1)
            plt.plot(x_1, I_numerical, label = f"{n_photon} fotoner")
        plt.title("Numerical (scaled) intensity for different number of photons compared to the analytical solution.")
        plt.legend()
        plt.grid()
        plt.show()
        st.pyplot(fig1)

        # Figure 2 showing the numerical approximation for different steplengths
        fig2 = plt.figure(figsize = (9,7))
        plt.plot(x_1, I_analytical, label = "Analytical", linestyle = 'dotted', linewidth = 4)
        n_steps_1_2 = [100, 50, 25, 10, 5, 2] # Different number of steps to compare
        dx1_2 = width_1 / np.array(n_steps_1_2)

        # Simulate 100,000 photons through the material with the steplengths definied above
        for i, n_step in enumerate(n_steps_1_2):
            x1_2 = np.linspace(0, width_1, n_step + 1) # Need new x-axis for each simulation
            my1_2 = 0.1 * np.ones(n_step) # Dimension of my changes with number of steps
            I = sim.simulate_photons(n_photons_1, n_step, width_1, my1_2) # Do the simulation
            
            plt.plot(x1_2, I, label = f"$\Delta x$ = {dx1_2[i]} cm")

        plt.title("Monte Carlo-method with different $\Delta$x")
        plt.xlabel("x [cm]")
        plt.ylabel("$I/I_0$")
        plt.legend()
        plt.grid()
        st.pyplot(fig2)


if __name__ == "__main__":
    main()