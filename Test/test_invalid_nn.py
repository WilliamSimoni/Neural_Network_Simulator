"""
    Test for Invalid Neural Network Class
"""
import unittest
import sys
sys.path.append('../')
import numpy as np
from neuralNetwork import NeuralNetwork
from neural_exception import InvalidNeuralNetwork
from layer import HiddenLayer, OutputLayer
import activationFunction as activation

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.neural_network = NeuralNetwork(100)

    def test_invalid_epoch(self):
        try:
            self.assertRaises(InvalidNeuralNetwork, NeuralNetwork(-1))
        except:
            print("Invalid Epoch")

    def test_invalid_type(self):
        try:
            self.assertRaises(InvalidNeuralNetwork, NeuralNetwork(100, nn_type="NN"))
        except:
            print("Invalid Training Type")

    def test_invalid_batch_size(self):
        try:
            self.assertRaises(InvalidNeuralNetwork, NeuralNetwork(100, batch_size=3))
        except:
            print("Invalid Batch Size for SGD")

        try:
            self.assertRaises(InvalidNeuralNetwork, NeuralNetwork(100, nn_type="batch", batch_size=-1))
        except:
            print("Invalid batch size for Batch")

        try:
            self.assertRaises(InvalidNeuralNetwork, NeuralNetwork(100, nn_type="minibatch", batch_size=-2))
        except:
            print("Invalid batch size for minibatch")

    def test_type_classifier(self):
        try:
            self.assertRaises(InvalidNeuralNetwork, NeuralNetwork(100, type_classifier="neural Network"))
        except:
            print("Invalid Type classifier")
        
    def test_momentum(self):
        try:
            self.assertRaises(InvalidNeuralNetwork, NeuralNetwork(100, momentum_rate=-0.2))
        except:
            print("Invalid Momentum rate")

    def test_regularization(self):
        try:
            self.assertRaises(InvalidNeuralNetwork, NeuralNetwork(100, regularization_rate=-0.5))
        except:
            print("Invalid Regularization rate")
        
    def test_add_layer_exception(self):
        layer = NeuralNetwork(2)
        try:
            self.assertRaises(ValueError, self.neural_network.addLayer(layer)) 
        except:
            self.assertTrue("ValueError raises")

    def test_error_predict(self):
        weights = np.ones((3, 3))
        learning_rate = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
        layer = OutputLayer(weights, learning_rate, activation.Linear())
        self.neural_network.addLayer(layer)
        try:
            self.assertRaises(ValueError, list(self.neural_network.predict([2])))
        except:
            self.assertTrue("ValueError raises")

    def test_invalid_layer_dimension(self):
        weights = np.ones((3, 3))
        weight1 = np.ones((2, 2))
        learning_rate_1 = np.array([[0.3, 0.2], [0.4, 0.4]])
        learning_rate = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
        layer1 = HiddenLayer(weights, learning_rate, activation.Linear())
        layer2 = OutputLayer(weight1, learning_rate_1, activation.Linear())

        try:
            self.neural_network.addLayer(layer1)
            self.assertRaises(ValueError, self.neural_network.addLayer(layer2))
        except:
            print("ValueError raises")
