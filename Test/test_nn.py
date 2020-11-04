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
        self.neural_network = NeuralNetwork(100)
    
    def test_constructor(self):    
        self.assertEqual(self.neural_network.max_epochs, 100)
        self.assertEqual(len(self.neural_network.layers), 0)

    def test_add_layer(self):
        weights = np.ones((3, 3))
        learning_rate = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
        layer = OutputLayer(weights, learning_rate, activation.Linear())
        self.neural_network.addLayer(layer)
        self.assertEqual(len(self.neural_network.layers), 1)
        
    def test_add_layer_exception(self):
        layer = NeuralNetwork(2)
        try:
            self.assertRaises(ValueError, self.neural_network.addLayer(layer)) 
        except:
            self.assertTrue("ValueError raises")
    
    def test_predict(self):
        weights = np.ones((3, 3))
        learning_rate = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
        layer = OutputLayer(weights, learning_rate, activation.Linear())
        self.neural_network.addLayer(layer)
        self.assertListEqual(list(self.neural_network.predict([1, 2])), [4, 4, 4])

    def test_error_predict(self):
        weights = np.ones((3, 3))
        learning_rate = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
        layer = OutputLayer(weights, learning_rate, activation.Linear())
        self.neural_network.addLayer(layer)
        try:
            self.assertRaises(ValueError, list(self.neural_network.predict([2])))
        except:
            self.assertTrue("ValueError raises")

    

if __name__ == '__main__':
    unittest.main()