import unittest
import simulation_functions as sf
from random import Random
random = Random(123)
import numpy as np

class Test_simulate_photons(unittest.TestCase):
    """
    Class for testing the simulate photons function
    """
    def test_intensity_decreasing(self):
        # Setting up parameters
        n_photons = random.randint(10000, 100000)
        n_steps = random.randint(100, 1000)
        width = random.randint(5,20)
        mu = random.random() * np.ones(n_steps)
        intensity = sf.simulate_photons(n_photons, n_steps, width, mu)

        # Check that intensity is decreasing
        for i in range(len(intensity)-1):
            self.assertGreaterEqual(intensity[i], intensity[i+1])


if __name__ == '__main__':
    unittest.main()
