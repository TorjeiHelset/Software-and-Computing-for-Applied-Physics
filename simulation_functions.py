from numba import jit #just in time compilation - faster code
import numpy as np


@jit(nopython=True) # just-in-time compilation
def simulate_photons(n_photons, n_steps, width, my):
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
    
    
    dx = width / n_steps
    lengths = np.zeros(n_photons) # Each element i is the lenght (cm) photon i reached
    p = my * dx # Pobabilities of being scattered/absorbed at the different steps           
    r = np.random.rand(n_photons, n_steps) # Random numbers for comparing with p
    
    for i in range(n_photons):    
        # Simulate each photon seperately      
        length = width # If not scattered/absorbed this won't be overwritten
        for j in range(n_steps):
            # Going through each step in the material
            if r[i, j] < p[j]:
                # If true the photon was absorbed during step j
                length = dx * j # The length the photon reached
                break
        lengths[i] = length # Storing the length photon i reached
    
    I = np.zeros(n_steps + 1) # Contains the relative intensity at each gridpoint (including start/end)

    I[0] = 1 # First intensity equal to 1
    for i in range(n_steps):
        # Calculating the intensity after each step
        nFotoner_i = np.sum(lengths == dx * i) # The number of photons absorbed in this step
        I[i+1] = I[i] - nFotoner_i / n_photons # Removing the percentage of photons absorbed from intensity
    
    return I



@jit(nopython=True) # just-in-time compilation
def simulate_photons_detector(n_photons, n_steps, width, my):
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
    
    
    dx = width / n_steps
    p = my * dx # Pobabilities of being scattered/absorbed at the different steps           
    r = np.random.rand(n_photons, n_steps) # Random numbers for comparing with p
    n_hits = 0 # Stores the number of photons hitting the detector
    
    for i in range(n_photons):    
        # Simulate each photon seperately      
        length = width # If not scattered/absorbed this won't be overwritten
        for j in range(n_steps):
            # Going through each step in the material
            if r[i, j] < p[j]:
                # If true the photon was absorbed during step j
                length = dx * j # The length the photon reached
                break
        if length == width:
            # Detector was reached
            n_hits += 1

    return n_hits / n_photons


@jit(nopython = True)
def simulate_photons_3D(n_photons, object, widths):
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
        
        Output:
            xy_plane (1d float array): intensity at detector at each point in xy-plane
            yz_plane (1d float array): intensity at detector at each point in yz-plane
            xz_plane (1d float array): intensity at detector at each point in xz-plane
    """
    
    nX, nY, nZ = object.shape # Get the number of gridpoint in each direction (dimension of object)
    xy_plane, yz_plane, xz_plane = np.zeros((nX, nY)), np.zeros((nY, nZ)), np.zeros((nX, nZ)) # Creating the planes
    
    # Going through the xy plane
    for x in range(nX):
        for y in range(nY):
            my = object[x, y, :] # Getting attenuation coefficients
            xy_plane[x, y] = simulate_photons_detector(n_photons, nZ, widths[2], my)
    
    # Going through the yz plane
    for y in range(nY):
        for z in range(nZ):
            my = object[:, y, z] # Getting attenuation coefficients
            yz_plane[y, z] = simulate_photons_detector(n_photons, nX, widths[0], my)
            
    # Going through the xz plane
    for x in range(nX):
        for z in range(nZ):
            my = object[x, :, z] #henter riktig my
            xz_plane[x, z] = simulate_photons_detector(n_photons, nY, widths[1], my)
    
    return xy_plane, yz_plane, xz_plane