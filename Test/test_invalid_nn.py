"""
    Test for Invalid Neural Network Class
"""
import unittest
import sys
sys.path.append('../')
import numpy as np
from neural_network import NeuralNetwork
from neural_exception import InvalidNeuralNetwork
from layer import HiddenLayer, OutputLayer
import activation_function as activation

class TestInvalidNeuralNetwork(unittest.TestCase):
    """
        Test NN with invalid results/initialization
    """

    def setUp(self):
        self.neural_network = NeuralNetwork(100)

    def test_invalid_epoch(self):
        try:
            self.assertRaises(InvalidNeuralNetwork, NeuralNetwork(-1))
        except:
            pass

    def test_invalid_type(self):
        try:
            self.assertRaises(InvalidNeuralNetwork, NeuralNetwork(100, nn_type="NN"))
        except:
            pass

    def test_invalid_batch_size(self):
        try:
            self.assertRaises(InvalidNeuralNetwork, NeuralNetwork(100, batch_size=3))
            self.assertRaises(InvalidNeuralNetwork, NeuralNetwork(100, nn_type="batch", batch_size=-1))
            self.assertRaises(InvalidNeuralNetwork, NeuralNetwork(100, nn_type="minibatch", batch_size=-2))
        except:
            pass


    def test_type_classifier(self):
        try:
            self.assertRaises(InvalidNeuralNetwork, NeuralNetwork(100, type_classifier="neural Network"))
        except:
            pass
        
    def test_momentum(self):
        try:
            self.assertRaises(InvalidNeuralNetwork, NeuralNetwork(100, momentum_rate=-0.2))
        except:
            pass

    def test_regularization(self):
        try:
            self.assertRaises(InvalidNeuralNetwork, NeuralNetwork(100, regularization_rate=-0.5))
        except:
            pass
        
    def test_add_layer_exception(self):
        layer = NeuralNetwork(2)
        try:
            self.assertRaises(ValueError, self.neural_network.add_layer(layer)) 
        except:
            pass

    def test_error_predict(self):
        weights = np.ones((3, 3))
        learning_rate = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
        layer = OutputLayer(weights, learning_rate, activation.Linear())
        self.neural_network.add_layer(layer)
        try:
            self.assertRaises(ValueError, list(self.neural_network.predict([2])))
        except:
            pass

    def test_invalid_layer_dimension(self):
        weights = np.ones((3, 3))
        weight1 = np.ones((2, 2))
        learning_rate_1 = np.array([[0.3, 0.2], [0.4, 0.4]])
        learning_rate = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
        layer1 = HiddenLayer(weights, learning_rate, activation.Linear())
        layer2 = OutputLayer(weight1, learning_rate_1, activation.Linear())

        try:
            self.neural_network.add_layer(layer1)
            self.assertRaises(ValueError, self.neural_network.add_layer(layer2))
        except:
            pass

    def test_invalid_classification_predict(self):
        """
            Test an invalid classification of an input sample in a Classification NN
        """
        pass

    def test_invalid_regression_predict(self):
        """
            Test an invalid regression predict of an input sample in a Regression NN
        """
        pass