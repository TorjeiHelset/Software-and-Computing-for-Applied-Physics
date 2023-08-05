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
    
    def test_intensity_proper_value(self):
        # Run simulation 
        intensity = sf.simulate_photons(self.n_photons, self.n_steps, self.width, self.mu)

        # Check that all intensities are between 0 and 1
        for i in intensity:
            self.assertLessEqual(i - 1e-5, 1)
            self.assertGreaterEqual(i + 1e-5, 0)

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


class simulate_photons_detector(unittest.TestCase):
    """
    Class for testing the simulate photons detector function
    """
    def setUp(self):
        # Setting up parameters
        self.n_photons = random.randint(10000, 100000)
        self.n_steps = random.randint(100, 1000)
        self.width = random.randint(5,20)
        self.mu = random.random() * np.ones(self.n_steps)
    
    def test_intensity_proper_value(self):
        # Run simulation 
        intensity = sf.simulate_photons_detector(self.n_photons, self.n_steps, self.width, self.mu)

        # Check that all intensities are between 0 and 1
        self.assertLessEqual(intensity - 1e-5, 1)
        self.assertGreaterEqual(intensity + 1e-5, 0)

    def test_comparable_result(self):
        # Checking that the intensity at the decode is similar to the one from other simulation
        # approach

        # Run simulation for both approaches
        old_approach = sf.simulate_photons(self.n_photons, self.n_steps, self.width, self.mu)
        target = old_approach[-1]
        end_intensity = sf.simulate_photons_detector(self.n_photons, self.n_steps, self.width, self.mu)
        
        # Checking that the two intensities are "close enough"
        self.assertAlmostEqual(target, end_intensity, delta = 0.001)

if __name__ == '__main__':
    unittest.main()
