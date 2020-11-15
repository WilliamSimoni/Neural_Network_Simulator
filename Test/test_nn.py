"""
    Test for Neural Network Class
"""
import unittest
import sys
sys.path.append('../')
import numpy as np
from neuralNetwork import NeuralNetwork
from layer import HiddenLayer, OutputLayer
import activationFunction as activation

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.neural_network = NeuralNetwork(100, momentum_rate=0.2, regularization_rate=0.5)
        self.nn2 = NeuralNetwork(100, nn_type="batch", batch_size=5, type_classifier="regression")
        self.nn3 = NeuralNetwork(100, nn_type="minibatch", batch_size=4)


    def test_constructor(self):    
        self.assertEqual(self.neural_network.max_epochs, 100)
        self.assertEqual(len(self.neural_network.layers), 0)
        self.assertEqual(0.2, self.neural_network.momentum_rate)
        self.assertEqual(0.5, self.neural_network.regularization_rate)
        self.assertEqual("SGD", self.neural_network.type)
        self.assertEqual(1, self.neural_network.batch_size)
        self.assertEqual("classification", self.neural_network.type_classifier)
    
    def test_constructor_different_type(self):
        self.assertEqual("batch", self.nn2.type)
        self.assertEqual(5, self.nn2.batch_size)
        self.assertEqual("regression", self.nn2.type_classifier)
        self.assertEqual("minibatch", self.nn3.type)
        self.assertEqual(4, self.nn3.batch_size)

    def test_add_layer(self):
        weights = np.ones((3, 3))
        learning_rate = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
        layer = OutputLayer(weights, learning_rate, activation.Linear())
        self.neural_network.addLayer(layer)
        self.assertEqual(len(self.neural_network.layers), 1)
            
    def test_predict(self):
        weights = np.ones((3, 3))
        learning_rate = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
        layer = OutputLayer(weights, learning_rate, activation.Linear())
        self.neural_network.addLayer(layer)
        self.assertListEqual(list(self.neural_network.predict([1, 2])), [4, 4, 4])


if __name__ == '__main__':
    unittest.main()