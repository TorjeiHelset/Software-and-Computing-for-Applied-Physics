import unittest
import simulation_functions as sf
from random import Random
random = Random(123)
import numpy as np

class Test_simulate_photons(unittest.TestCase):
    """
    Class for testing the simulate photons function
    """
    def setUp(self):
        # Setting up parameters
        self.n_photons = random.randint(10000, 100000)
        self.n_steps = random.randint(100, 1000)
        self.width = random.randint(5,20)
        self.mu = random.random() * np.ones(self.n_steps)

    def test_intensity_decreasing(self):
        # Run simulation
        intensity = sf.simulate_photons(self.n_photons, self.n_steps, self.width, self.mu)

        # Check that intensity is decreasing
        for i in range(len(intensity)-1):
            self.assertGreaterEqual(intensity[i], intensity[i+1])
        
    def test_intensity_length_correct(self):
        # Run simulation
        intensity = sf.simulate_photons(self.n_photons, self.n_steps, self.width, self.mu)

        # Check that the outputed intensity has length equal to the number of steps + 1 (including start/end)
        self.assertEqual(len(intensity), self.n_steps + 1)


if __name__ == '__main__':
    unittest.main()
