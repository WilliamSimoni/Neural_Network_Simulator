"""
    Test Activation function module
"""
import unittest
import numpy as np
import activationFunction as af


class TestActivation(unittest.TestCase):
    """
        TestActivation test ActivationFunction module 
    """

    def test_linear_activation(self):
        """
            Test linear Activation function 
        """
        self.assertEqual([0.5, 0.6], af.Linear().output([0.5, 0.6]))
        self.assertEqual([1, 1], list(
            af.Linear().derivative(np.array([0.5, 0.6]))))

    def test_sigmoid_activation(self):
        """
            Test Sigmoid Activation function
        """
        self.assertEqual([0.5, 0.5], list(
            af.Sigmoid().output(np.array([0, 0]))))
        self.assertEqual([0.25, 0.25], list(
            af.Sigmoid().derivative(np.array([0, 0]))))

    def test_tanh_activation(self):
        """
            Test Tanh Activation function
        """
        self.assertEqual([0.099668, 0.099668], list(
            af.TanH().output(np.array([0.1, 0.1]))))
        self.assertEqual([0.990066, 0.990066], list(
            af.TanH().derivative(np.array([0.1, 0.1]))))

    def test_relu_activation(self):
        """
            Test Relu Activation function
        """
        self.assertEqual([0, 0.5], list(
            af.Relu().output(np.array([-0.5, 0.5]))))
        self.assertEqual([0, 1], list(
            af.Relu().derivative(np.array([-0.5, 0.5]))))

    def test_leaky_relu_activation(self):
        """
            Test Leaky Relu Activation function
        """
        self.assertEqual(
            [-0.0050, 0.5000], list(af.LeakyRelu(0.01).output(np.array([-0.5, 0.5]))))
        self.assertEqual([0.01, 1], list(af.LeakyRelu(
            0.01).derivative(np.array([-0.5, 0.5]))))

    def test_softplus_activation(self):
        """
            Test Softplus Activation function
        """
        self.assertEqual(
            [0.474077, 0.974077], list(af.SoftPlus().output(np.array([-0.5, 0.5]))))
        self.assertEqual([0.377541, 0.622459], list(
            af.LeakyRelu().derivative(np.array([-0.5, 0.5]))))
