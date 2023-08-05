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


class Test_simulate_photons_detector(unittest.TestCase):
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
        self.assertAlmostEqual(target, end_intensity, delta = 0.005)

class Test_simulate_photons_3D(unittest.TestCase):
    """
    Class for testing the simulate photons detector function
    """
    def setUp(self):
        # Setting up parameters
        self.n_photons = random.randint(100, 1000)
        self.widths = np.array([random.randint(5,20) for _ in range(3)])
        coinflip = random.random()
        energylevel = random.randint(1,3)

        if coinflip > 0.5:
            objects = np.array([np.load("data/object1_20keV.npy"), 
                                np.load("data/object1_50keV.npy"), 
                                np.load("data/object1_100keV.npy")])
        else:
            objects = np.array([np.load("data/object2_25keV.npy"),
                                np.load("data/object2_50keV.npy"),
                                np.load("data/object2_75keV.npy")])
        
        if energylevel == 1:
            self.object = objects[0]
        elif energylevel == 2:
            self.object = objects[1]
        else:
            self.object = objects[2]
        self.nX, self.nY, self.nZ = self.object.shape

    def test_intensities_proper_values(self):
        # Run simulation
        xy, yz, xz = sf.simulate_photons_3D(self.n_photons, self.object, self.widths)

        # Check that all values between 0 and 1
        for x in range(self.nX):
            for y in range(self.nY):
                self.assertLessEqual(xy[x,y] - 1e-5, 1)
                self.assertGreaterEqual(xy[x,y] + 1e-5, 0)
        
        # Going through the yz plane
        for y in range(self.nY):
            for z in range(self.nZ):
                self.assertLessEqual(yz[y,z] - 1e-5, 1)
                self.assertGreaterEqual(yz[y,z] + 1e-5, 0)
                
        # Going through the xz plane
        for x in range(self.nX):
            for z in range(self.nZ):
                self.assertLessEqual(xz[x,z] - 1e-5, 1)
                self.assertGreaterEqual(xz[x,z] + 1e-5, 0)
    
    def test_output_correct_shapes(self):
        # Run simulation and getting output shapes
        xy, yz, xz = sf.simulate_photons_3D(self.n_photons, self.object, self.widths)
        nx1, ny1 = xy.shape
        ny2, nz2 = yz.shape
        nx3, nz3 = xz.shape

        # Checking that all shapes allign
        self.assertEqual(self.nX, nx1)
        self.assertEqual(self.nX, nx3)

        self.assertEqual(self.nY, ny1)
        self.assertEqual(self.nY, ny2)

        self.assertEqual(self.nZ, nz2)
        self.assertEqual(self.nZ, nz3)


if __name__ == '__main__':
    unittest.main()
