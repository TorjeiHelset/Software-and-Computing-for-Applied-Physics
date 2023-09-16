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