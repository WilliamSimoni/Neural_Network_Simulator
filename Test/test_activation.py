import unittest
import numpy as np
import activationFunction as af

class TestActivation(unittest.TestCase):
    """
        TestActivation test ActivationFunction module 
    """

    def test_linear_activation(self):
        self.assertEqual([0.5, 0.6], af.Linear().output([0.5, 0.6]))
        self.assertEqual([1, 1], af.Linear().derivative(np.array([0.5, 0.6])))

    def test_sigmoid_activation(self):
        self.assertEqual([0.5, 0.5], af.Sigmoid().output(np.array([0, 0])))
        self.assertEqual([0.25, 0.25], af.Sigmoid().derivative(np.array([0, 0])))

    def test_tanh_activation(self):
        self.assertEqual

    def test_relu_activation(self):
        pass

    def test_leaky_relu_activation(self):
        pass

    def test_softplus_activation(self):
        pass