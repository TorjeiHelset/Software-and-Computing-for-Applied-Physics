from numba import jit #just in time compilation - faster code
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display # For tables
import tabulate # Tables

@jit(nopython=True) # just-in-time compilation
def simulate_single_photon(width, n_steps, r, p, dx):
    '''
    Simulates one photon passing through a material of given width using n_steps.
    For each step there is a probability p of being absorbed.

        Parameters:
            width (float) :             Width of material
            n_steps (int) :             Number of steps in simulation
            r (1D array of float) :     Array of numbers sampled from uniform distribution
            p (float) :                 Probability of being absorbed at each step
            dx (float) :                Length of each step

        Output:
            length :                    Length reached by photon in material
    '''

    length = width # If not scattered/absorbed this won't be overwritten
    for j in range(n_steps):
        # Going through each step in the material
        if r[j] < p[j]:
            # If true the photon was absorbed during step j
            length = dx * j # The length the photon reached
            break
    return length
    


@jit(nopython=True) # just-in-time compilation
def simulate_photons(n_photons, n_steps, width, my, set_seed=123):
    """
    Approximates the intensity of a photon beam moving through a material evaluated at different steps.
    The approximation is based on a Monte Carlo method using the fact that for each distance x to x + Delta x
    there is a probability p that the photon either will scatter or be absorved. This probability is equal to
    the length moved and the attenuation coefficient. A random number is sampled at each step and compared to this 
    probability to determine if the photon will keep on moving or not.

        Parameters:
            n_photons (int):        number of photons
            n_steps (int):          number of discrete steps taken through the material
            width (float):          the width of the material (cm)
            my (1D float array):    the attenuation coefficient of the material at each step
            set_seed (int):         seed to get reproducable results
        
        Output:
            I (1D float array):     the intensity at each step relative to initial intensity

        Raise:
            ValueError if n_steps is not equal to length of my
            ValueError if n_photons is smaller than 1
            ValueError if width is negative
    """

    if n_steps != len(my):
        raise ValueError('The number of steps must be equal to the number of attenuation coefficients.')
    if n_photons < 1:
        raise ValueError('The number of photons must be positive.')
    if width <= 0:
        raise ValueError('The width must be positive.')
    
    np.random.seed(set_seed)

    dx = width / n_steps
    lengths = np.zeros(n_photons) # Each element i is the lenght (cm) photon i reached
    p = my * dx # Pobabilities of being scattered/absorbed at the different steps           
    r = np.random.rand(n_photons, n_steps) # Random numbers for comparing with p
    
    for i in range(n_photons):    
        # Simulate each photon seperately      
        lengths[i] = simulate_single_photon(width, n_steps, r[i,:], p, dx) # Storing the length photon i reached
    
    I = np.zeros(n_steps + 1) # Contains the relative intensity at each gridpoint (including start/end)

    I[0] = 1 # First intensity equal to 1
    for i in range(n_steps):
        # Calculating the intensity after each step
        nFotoner_i = np.sum(lengths == dx * i) # The number of photons absorbed in this step
        I[i+1] = I[i] - nFotoner_i / n_photons # Removing the percentage of photons absorbed from intensity
    
    return I



@jit(nopython=True) # just-in-time compilation
def simulate_photons_detector(n_photons, n_steps, width, my, set_seed=123):
    """
    Approximates the intensity of a photon beam after having moved through a material evaluated.
    The approximation is based on a Monte Carlo method using the fact that for each distance x to x + Delta x
    there is a probability p that the photon either will scatter or be absorved. This probability is equal to
    the length moved and the attenuation coefficient. A random number is sampled at each step and compared to this 
    probability to determine if the photon will keep on moving or not.

        Parameters:
            n_photons (int):        number of photons
            n_steps (int):          number of discrete steps taken through the material
            width (float):          the width of the material (cm)
            my (1D float array):    the attenuation coefficient of the material at each step
            set_seed (int):         seed to get reproducable results
        
        Output:
            n_hits / n_photons:     the percentage of photons reaching the detector

        Raise:
            ValueError if n_steps is not equal to length of my
            ValueError if n_photons is smaller than 1
            ValueError if width is negative
    """

    if n_steps != len(my):
        raise ValueError('The number of steps must be equal to the number of attenuation coefficients.')
    if n_photons < 1:
        raise ValueError('The number of photons must be positive.')
    if width <= 0:
        raise ValueError('The width must be positive.')
    
    np.random.seed(set_seed)

    dx = width / n_steps
    p = my * dx # Pobabilities of being scattered/absorbed at the different steps           
    r = np.random.rand(n_photons, n_steps) # Random numbers for comparing with p
    n_hits = 0 # Stores the number of photons hitting the detector
    
    for i in range(n_photons):    
        # Simulate each photon seperately      
        length = simulate_single_photon(width, n_steps, r[i,:], p, dx)
        if length == width:
            # Detector was reached
            n_hits += 1

    return n_hits / n_photons


@jit(nopython = True)
def simulate_photons_3D(n_photons, object, widths, set_seed=123):
    """
    Takes in a 3d matrix "object" describing the attenuation coefficient of a material
    at different grid points. Approximates the intensity of a photon beam moving through
    every point on the three planes. Returns one 2D array of intensities at the detector for
    each of the three planes. The simulation is performed by simulate_photons_detector.

        Parameters:
            n_photons (int):         number of photons
            object (3D float array): matrix of attenuation coefficients at grid points
            width (1d float array):  the widths of the material (cm) in each direction
            energy (int):            energy of the photon beam to be used in the simulation
            set_seed (int):         seed to get reproducable results
        
        Output:
            xy_plane (2d float array): intensity at detector at each point in xy-plane
            yz_plane (2d float array): intensity at detector at each point in yz-plane
            xz_plane (2d float array): intensity at detector at each point in xz-plane
    """
    
    nX, nY, nZ = object.shape # Get the number of gridpoint in each direction (dimension of object)
    xy_plane, yz_plane, xz_plane = np.zeros((nX, nY)), np.zeros((nY, nZ)), np.zeros((nX, nZ)) # Creating the planes
    
    # Going through the xy plane
    for x in range(nX):
        for y in range(nY):
            my = object[x, y, :] # Getting attenuation coefficients
            xy_plane[x, y] = simulate_photons_detector(n_photons, nZ, widths[2], my, set_seed)
    
    # Going through the yz plane
    for y in range(nY):
        for z in range(nZ):
            my = object[:, y, z] # Getting attenuation coefficients
            yz_plane[y, z] = simulate_photons_detector(n_photons, nX, widths[0], my, set_seed)
            
    # Going through the xz plane
    for x in range(nX):
        for z in range(nZ):
            my = object[x, :, z] #henter riktig my
            xz_plane[x, z] = simulate_photons_detector(n_photons, nY, widths[1], my)
    
    return xy_plane, yz_plane, xz_plane


def compare_different_n_photon(n_photons, n_steps, width, my):
    '''
    Compares results from Monte-Carlo simulation with analytical solution for different 
    choices of number of photons.

    Parameters:
        n_photons (1D array of int) :   Array of different number of photons
        n_steps (int) :                 Number of steps to be used in simulation
        width (float) :                 Width of material (cm)
        my (1D array of foat) :         List of attenuation coefficients at each step
    '''
    x_1 = np.linspace(0, width, n_steps + 1)

    # Analytical solution given the parameters:
    I_analytical = np.exp(-my[0] * x_1) # For this part we assume attenuation coefficient to be constant

    fig1 = plt.figure()
    plt.plot(x_1, I_analytical, label = 'Analytical', linestyle = '--', color = 'black', linewidth = 1)

    for i, n_photon in enumerate(n_photons):
        # Going through the different numbers of photons
        I_numerical = simulate_photons(n_photon, n_steps, width, my)
        plt.plot(x_1, I_numerical, label = f"{n_photon} photons")
    plt.title("Numerical (scaled) intensity for different number of photons compared to the analytical solution.")
    plt.legend()
    plt.show()


def compare_different_n_table(n_photons, n_steps, width, my):
    '''
    Compares the results from Monte-Carlo simulation to the analytical ones by running simulation 1000
    times for each choice of number of photons. Calculates mean error and standard deviation of error and
    summarizing in table.

        Parameters:
            n_photons (1D array of int) :   Array of different number of photons
            n_steps (int) :                 Number of steps to be used in simulation
            width (float) :                 Width of material (cm)
            my (1D array of foat) :         List of attenuation coefficients at each step

    '''
    x_1 = np.linspace(0, width, n_steps + 1)
    I_analytical = np.exp(-my[0] * x_1) # For this part we assume attenuation coefficient to be constant
    
    table = []
    for i, n_photon in enumerate(n_photons):
        distance = np.zeros(n_steps + 1)
        for j in range(1000):
            I = simulate_photons(n_photon, n_steps, width, my)
            
            distance += abs(I_analytical - I) # Summing up the total distances of all the runs
        
        # Calculate averages and standard deviations
        mean_error = np.mean(distance/1000) 
        std_error = np.std(distance/1000)
        
        # Adding result to table for easier
        table.append([f"{n_photon}", round(mean_error, 4), round(std_error, 4)])
        
    # Using library tabulate to get a good looking table
    # Changing the html code to get a centered table
    display(HTML('<table width="80%" style="margin: 0px auto;"><thead><tr><th style="text-align: center;">' 
        + tabulate.tabulate(table, headers=["Photons", "Mean error I/I0", "Standard deviation I/I0"], 
                                tablefmt="html")[52:-8] + "</table>"))
    

def compare_differnt_steplengths(n_photons, n_steps, width, my):
    '''
    Compares results from Monte-Carlo simulation with analytical solution for different 
    choices of number of steplenghts.

        Parameters:
            n_photons (int) :               Number of photons to be used in simulation
            n_steps (1D array of int) :     List of different steps to be used in simulation
            width (float) :                 Width of material (cm)
            my (Foat) :                     Attenuation coefficient of material

    '''
    x_1 = np.linspace(0, width, n_steps[-1] + 1)
    I_analytical = np.exp(-my * x_1) # For this part we assume attenuation coefficient to be constant
    fig2 = plt.figure()
    plt.plot(x_1, I_analytical, label = "Analytical", linestyle = 'dotted', linewidth = 4)
    dx = width / np.array(n_steps)

    # Simulate 100,000 photons through the material with the steplengths definied above
    for i, n_step in enumerate(n_steps):
        x1_2 = np.linspace(0, width, n_step + 1) # Need new x-axis for each simulation
        my_dim = my * np.ones(n_step) # Dimension of my changes with number of steps
        I = simulate_photons(n_photons, n_step, width, my_dim) # Do the simulation
        
        plt.plot(x1_2, I, label = f"$\Delta x$ = {dx[i]} cm")

    plt.title("Monte Carlo-method with different $\Delta$x"); plt.xlabel("x [cm]"); plt.ylabel("$I/I_0$"); plt.legend()
    plt.show()


def simulate_photons_bone_and_tissue(energy_tissue, energy_bone, energy_higher, energy_lower, mu_tissue, mu_bone,
                                     n_steps, width, n_photons):
    '''
    Simulates photon beams going through only tissue and material going through tissue and bone. Photon beams of different 
    energies will be sent through the two materials.

        Parameters :
            energy_tissue (1D array of float) :     Array of energies where there is data on the attenuation coefficient for tissue
            energy_bone (1D array of float) :       Array of energies where there is data on the attenuation coefficient for bone
            energy_higher (float) :                 Upper bound of energy to be considered
            energy_lower (float) :                  Lower bound of energy to be considered
            mu_tissue (1D array of float) :         Array of attenuation coefficient of tissue for different energies
            mu_bone (1D array of float) :           Array of attenuation coefficient of bone for different energies
            n_steps (int) :                         Number of steps to simulate
            width (float) :                         Width of material
            n_photons (int) :                       Number of photons to be used in material

        Out:
            energies (1D array of float) :          Array of energies between lower and upper bound
            I_tissue_detector (1D array of float) : Intensity of photon beams of different energies after passing through tissue
            I_bone_detector (1D array of float) :   Intensity of photon beams of different energies after passing through tissue and bone
            I_after_tissue (1D array of float) :    Intensity of beam after passing through section of only tissue
            I_after_bone (1D array of float) :      Intensity of beam after passing through section of only tissue and section of only bone

    '''
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
        mu_tissue_i = mu_tissue[index_tissue] * np.ones(n_steps)
        mu_bone_i = mu_tissue_i.copy()
        
        # Filling in the bone region
        mu_bone_i[n_steps // width : 2 * n_steps // width] = mu_bone[index_bone]

        # Simulating n_photons_2 number of photons through the different materials
        I_tissue_detector[i] = simulate_photons(n_photons, n_steps, width, mu_tissue_i)[-1]
        I_bone_beam = simulate_photons(n_photons, n_steps, width, mu_bone_i)
        I_bone_detector[i] = I_bone_beam[-1]
        I_after_tissue[i] = I_bone_beam[n_steps//3]
        I_after_bone[i] = I_bone_beam[(2*n_steps)//3]

    return energies, I_tissue_detector, I_bone_detector, I_after_tissue, I_after_bone

def contrast(I1, I2):
    '''
    Calculates contrast between two phton beams

        Paramters:
            I1 (Array of float) :    Intensity of first photon beam
            I2 (Array of float) :    Intensity of second photon beam

        Out:
            Contrast (Array of float) : Contrast between the two photon beams

    '''

    contrast = np.zeros(len(I2))
    for i in range(len(I1)):
        contrast[i] = (I1[i] - I2[i]) / I1[i]

    return contrast

def calculate_absorpotion(I_tissue_detector, I_bone_detector, req_photons_tissue, I_after_tissue, I_after_bone,
                          energies, tissue_dens, bone_dens, width):
    # Calculating the number of photons that reached detector and each of the 
    # sections in the tissue and tissue/bone beams
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
    V = A * width #cm^3
    # Calculating the total absorbed dosages for the two beams
    d_tissue = n_absorbed_tissue * energies[1:] / (V * tissue_dens)
    d_bone = energies[1:] / V * ((n_in_tissue / (1/3 * tissue_dens))
                            +  (n_in_bone / (1/3 * bone_dens)) 
                            +  (n_absorbed_bone / (1/3 * tissue_dens))) 
    
    return d_tissue, d_bone

def read_in_objects(object_names):
    '''
    Reading in attenuation matrices from data folder

        Parameters:
            object_names (1D array of strings) :    List of filenames of file containing attenuation data

        Out:
            (1D array of 3D numpy array) :          List of 3D matrices containing attenuation of object at gridpoints
    '''
    return np.array([np.load(name) for name in object_names])

