import numpy as np # Numpy for handling arrays
import matplotlib.pyplot as plt # Matplotlib to do the plotting
import tabulate # Tables
from numba import jit #just in time compilation - faster code
import simulation_functions as sim
import generate_plots as gen

def simulation_1(width_1, n_steps_1, x_1, n_photons_1, my1):
    '''
    Runs a monte carlo simulation of photons beams passing through a material with attenuation
    coefficient my1. Compares the simulated results for different choices of photons and 
    steplengths with an analytical solution.

        Parameters:
            width_1 (int):            width of material
            n_steps_1 (int):          number of discrete steps taken through the material
            x_1 (1d float array):     array of depths in material 
            n_photons_1 (int):        Number of photons used in simulation
            my1 (1D float array):     Array of attenuation coefficient at different depths
        
        Output:
            fig1 (matplotlib figure): Plot showing difference between analytical and simulated results for different number of photons
            fig2 (matplotlib figure): Plot showing difference between analytical and simulated results for different steplengths
    '''
    # Analytical solution given the parameters:
    I_analytical = np.exp(-my1[0] * x_1) # For this part we assume attenuation coefficient to be constant

    # Figure 1 showing the numerical approximation for different choices of number of photons
    fig1 = plt.figure()
    plt.plot(x_1, I_analytical, label = 'Analytical', linestyle = '--', color = 'black', linewidth = 1)
    number_of_photons = [10, 50, 100, 1000, 10000, 100000]
    for i, n_photon in enumerate(number_of_photons):
        # Going through the different numbers of photons
        I_numerical = sim.simulate_photons(n_photon, n_steps_1, width_1, my1)
        plt.plot(x_1, I_numerical, label = f"{n_photon} fotoner")
    plt.title("Numerical (scaled) intensity for different number of photons compared to the analytical solution.")
    plt.legend()

    # Figure 2 showing the numerical approximation for different steplengths
    fig2 = plt.figure()
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
    return fig1, fig2

def simulation_2(width_2, n_steps_2, n_photons_2):
    '''
    Runs a monte carlo simulation of photons beams passing through a bone and tissue.
    Compares the contrast between the two materials, the number of required photons for 
    the beam to be detected and the absorbed dosage of the photon beams.

        Parameters:
            width_2 (int):            width of material
            n_steps_2 (int):          number of discrete steps taken through the material
            n_photons_2 (int):        Number of photons used in simulation
        
        Output:
            fig4 (matplotlib figure): Plot showing relative intensity at detector for different energies
            fig5 (matplotlib figure): Plot showing contrast between photon beams passing through bone and tissue
            fig6 (matplotlib figure): Plot showing the required number of photons for beam to be detected
            fig7 (matplotlib figure): Plot showing the absored dosage of beams passing through bone and tissue
            fig8 (matplotlib figure): Plot showing total absored dosage of photon beam
    '''
    tissue_dens = 1.02 #g/cm^3
    bone_dens = 1.92 #g/cm^3
    # reading in the datafiles for attenuation coefficients
    energy_tissue, mu_tissue = np.loadtxt("data/tissue.txt", delimiter=',', unpack=True)
    energy_bone, mu_bone = np.loadtxt("data/bone.txt", delimiter=',', unpack=True)
    # Multiplying with density to get correct attenuation coefficient
    mu_tissue *= tissue_dens
    mu_bone *= bone_dens
    energy_lower, energy_higher = 1e-2, 1e-1
    # Creating a list of energies that are contained in both bone and tissue data, and are within interval
    energies = energy_tissue[(energy_tissue <= energy_higher) & (energy_tissue >= energy_lower)] 

    # Two lists to store intensity at the detector for each energy level
    I_tissue_detector = np.zeros(len(energies)) #I1: kun vev
    I_bone_detector = np.zeros(len(energies)) #I2: vev-bein-vev

    # Also storing intensity for the second beam after tissue part and after bone part
    I_after_tissue = np.zeros(len(energies))
    I_after_bone = np.zeros(len(energies))

    for i, energy in enumerate(energies):
        # Collecting the correct attenuation coefficients for each of the two 
        index_tissue = np.where(energy_tissue == energy)[0][0] 
        index_bone = np.where(energy_bone == energy)[0][0]

        # Transforming the coefficient to lists of length equal to number of steps
        mu_tissue_i = mu_tissue[index_tissue] * np.ones(n_steps_2)
        mu_bone_i = mu_tissue_i.copy()
        
        # Filling in the bone region
        mu_bone_i[n_steps_2 // width_2 : 2 * n_steps_2 // width_2] = mu_bone[index_bone]

        # Simulating n_photons_2 number of photons through the different materials
        I_tissue_detector[i] = sim.simulate_photons(n_photons_2, n_steps_2, width_2, mu_tissue_i)[-1]
        I_bone_beam = sim.simulate_photons(n_photons_2, n_steps_2, width_2, mu_bone_i)
        I_bone_detector[i] = I_bone_beam[-1]
        I_after_tissue[i] = I_bone_beam[n_steps_2//3]
        I_after_bone[i] = I_bone_beam[(2*n_steps_2)//3]

    fig4 = plt.figure()
    plt.plot(energies*1000, I_tissue_detector, label = "$I_{tissue}$")
    plt.plot(energies*1000, I_bone_detector, label = "$I_{bone}$")
    plt.title("Relative intensity at detector for different energies.")
    plt.xlabel("Energi [keV]"); plt.ylabel("I/I$_0$"); plt.legend()

    fig5 = plt.figure()
    contrast = np.zeros(len(I_bone_detector))
    for i in range(len(I_tissue_detector)):
        contrast[i] = (I_tissue_detector[i] - I_bone_detector[i]) / I_tissue_detector[i]

    plt.plot(energies[1:]*1000, contrast[1:])
    plt.title("Contrast between $I_{tisse}$ og $I_{bone}$")
    plt.xlabel("Energi [keV]"); plt.ylabel("$I/I_0$")

    req_photons_tissue = 10 / (I_tissue_detector[1:] * energies[1:])
    req_photons_bone = 10 / (I_bone_detector[2:] * energies[2:])
    req_photons = 10 / energies[1:]

    fig6 = plt.figure()
    plt.semilogy(energies[1:]*1000, req_photons_tissue, label="Tisse")
    plt.semilogy(energies[2:]*1000, req_photons_bone, label = "Bone")
    plt.semilogy(energies[1:]*1000, req_photons, label = "Minimum")
    plt.title("Required number of photons for detection")
    plt.xlabel("Energy [keV]"); plt.ylabel("Number of photons")
    plt.legend()

    n_photon_tissue = I_tissue_detector[1:] * req_photons_tissue
    n_photon_bone = I_bone_detector[1:] * req_photons_tissue
    n_photon_after_tissue = I_after_tissue[1:] * req_photons_tissue # After the tissue part
    n_photon_after_bone = I_after_bone[1:] * req_photons_tissue # After bone part
    n_absorbed_tissue = req_photons_tissue - n_photon_tissue # Absorbed photons for the tissue beam

    # Calculating the number of photons absorbed in the second beam after each of the sections
    n_in_tissue = req_photons_tissue - n_photon_after_tissue
    n_in_bone = n_photon_after_tissue - n_photon_after_bone
    n_absorbed_bone = n_photon_after_bone - n_photon_bone

    A = 1 # Choosing area of 1 cm^2
    V = A * width_2 #cm^3
    # Calculating the total absorbed dosages for the two beams
    d_tissue = n_absorbed_tissue * energies[1:] / (V * tissue_dens)
    d_bone = energies[1:] / V * ((n_in_tissue / (1/3 * tissue_dens))
                            +  (n_in_bone / (1/3 * bone_dens)) 
                            +  (n_absorbed_bone / (1/3 * tissue_dens))) 
    
    fig7 = plt.figure()
    plt.semilogy(energies[1:]*1000, d_tissue, label = "Absorbed in tissue")
    plt.semilogy(energies[1:]*1000, d_bone, label  = "Absorbed in bone")
    plt.semilogy(energies[1:]*1000, d_tissue+d_bone, label = "Total absorbed dosage")
    plt.legend()
    plt.title("Absorbed dosage in tissue and bone")
    plt.xlabel("Energy [keV]")
    plt.ylabel("Energy per mass [MeV/g]")

    fig8, axes8 = plt.subplots()
    plt8 = axes8.semilogy(energies[1:]*1000, d_tissue+d_bone, label = "Total absorbed dosage")
    axes8_2 = axes8.twinx()
    plt8_2 = axes8_2.plot(energies[1:]*1000, contrast[1:], color = "red", label = "Contrast")
    axes8.set_xlabel("Energy [keV]")
    axes8.set_ylabel("Energy per mass [MeV/g]")
    axes8_2.set_ylabel("Contrast I/I0")

    ls = [l.get_label() for l in (plt8 + plt8_2)]
    axes8_2.legend(plt8 + plt8_2, ls)
    axes8.tick_params(axis="y", colors="blue")
    axes8_2.tick_params(axis="y", colors="red")

    return fig4, fig5, fig6, fig7, fig8

def simulation_3(n_photons_3, objects1, objects2, widths3_1, widhts3_2):
    '''
    Runs a monte carlo simulation of photons beams passing through a two different 3D objects.
    Sends photon beams through the objectives in all directions, and return plots showing 
    energy of beams having passed through the objects.

        Parameters:
            n_photons_3 (int):                  Number of photons used in simulation
            objects1 (list of 3D float array):  List of attenuation coefficient matrix for differnt energies for object 1
            objects2 (list of 3D float array):  List of attenuation coefficient matrix for differnt energies for object 2
            widths3_1 (1D array):               Length of each axis of object 1
            widths3_2 (1D array):               Length of each axis of object 2
        
        Output:
            fig9 (matplotlib figure):           Plot showing the intensity of photon beams having passed through object 1
            fig10 (matplotlib figure):          Plot showing the intensity of photon beams having passed through object 1
    '''
    # Simulating the photon beams through the two objects
    object_1, object_2 = [], []
    for i in range(3):
        object_1.append(sim.simulate_photons_3D(n_photons_3, objects1[i], widths3_1))
        object_2.append(sim.simulate_photons_3D(n_photons_3, objects2[i], widhts3_2))
        
    objects = [object_1, object_2]

    x_labs = ["x[cm]", "y[cm]", "x[cm]"]
    y_labs = ["y[cm]", "z[cm]", "z[cm]"]
    energy_labs = [["20keV", "50keV", "100keV"], ["25keV", "50keV", "75keV"]]
    extents = [[[0, 6.5, 0, 44.6], [0, 44.6, 0, 44.6], [0, 6.5, 0, 44.6]],
            [[0, 12, 0, 12], [0, 12, 0, 10], [0, 12, 0, 10]]]

    
    fig9, axes9 = plt.subplots(nrows = 3, ncols = 3, figsize= (9.5, 9))
    fig9.suptitle(f"Object 1")
    fig9.subplots_adjust(bottom = 0.1, top = 0.9, hspace = 0.4, wspace = 0.4)
    for j in range(3):
        for k  in range(3):
            axes9[j, k].grid(False)
            axes9[j,k].set_xlabel(x_labs[k])
            axes9[j,k].set_ylabel(y_labs[k])
            axes9[j,k].set_title(energy_labs[0][j])
            axes9[j, k].imshow(objects[0][j][k].T, extent = extents[0][k])
    
    fig10, axes10 = plt.subplots(nrows = 3, ncols = 3, figsize= (9.5, 9))
    fig10.suptitle(f"Object 2")
    fig10.subplots_adjust(bottom = 0.1, top = 0.9, hspace = 0.4, wspace = 0.4)
    for j in range(3):
        for k  in range(3):
            axes10[j, k].grid(False)
            axes10[j,k].set_xlabel(x_labs[k])
            axes10[j,k].set_ylabel(y_labs[k])
            axes10[j,k].set_title(energy_labs[1][j])
            axes10[j, k].imshow(objects[1][j][k].T, extent = extents[1][k])
    return fig9, fig10