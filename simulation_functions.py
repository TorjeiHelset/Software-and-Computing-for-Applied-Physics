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