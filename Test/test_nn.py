"""
    Test for Neural Network Class
"""
import unittest
import sys
sys.path.append("/home/bigboss98/Programming/Projects/Neural_Network_Simulator")
print(sys.path)
import numpy as np
from neural_network import NeuralNetwork
from layer import HiddenLayer, OutputLayer
import activation_function as activation

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        """
            Setup 3 neural network:
                -neural_network with classification and SGD as type
                -nn2 with batch and regression 
                -nn3 with minibatch and classification
        """
        self.neural_network = NeuralNetwork(100, 'euclidean_loss', '', momentum_rate=0.2, regularization_rate=0.5)
        self.nn2 = NeuralNetwork(100, 'euclidean_loss', '', nn_type="batch", batch_size=5, type_classifier="regression")
        self.nn3 = NeuralNetwork(100, 'euclidean_loss', '', nn_type="minibatch", batch_size=4)


    def test_constructor(self):
        """
            Test correctness of self.neural_network constructor
        """  
        self.assertEqual(self.neural_network.max_epochs, 100)
        self.assertEqual(len(self.neural_network.layers), 0)
        self.assertEqual(0.2, self.neural_network.momentum_rate)
        self.assertEqual(0.5, self.neural_network.regularization_rate)
        self.assertEqual("SGD", self.neural_network.type)
        self.assertEqual(1, self.neural_network.batch_size)
        self.assertEqual("classification", self.neural_network.type_classifier)
        self.assertEqual("euclidean_loss", self.neural_network.loss)
    
    def test_constructor_different_type(self):
        """
            Test correctness of self.nn2 and self.nn3 constructor
        """
        self.assertEqual("batch", self.nn2.type)
        self.assertEqual(5, self.nn2.batch_size)
        self.assertEqual("regression", self.nn2.type_classifier)
        self.assertEqual("minibatch", self.nn3.type)
        self.assertEqual(4, self.nn3.batch_size)

    def test_add_layer(self):
        """
            Test add to a different layer in NN
        """
        weights = np.ones((3, 3))
        learning_rate = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
        layer = OutputLayer(weights, learning_rate, activation.Linear())
        self.neural_network.addLayer(layer)
        self.assertEqual(len(self.neural_network.layers), 1)
            
    def test_regression_predict(self):
        """
            Test predict of an input sample using a regression NN
        """
        weights = np.ones((3, 3))
        learning_rate = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
        layer = OutputLayer(weights, learning_rate, activation.Linear())
        print(layer.get_num_input())
        self.nn2.add_layer(layer)
        self.assertListEqual(list(self.nn2.predict([1, 2])), [4, 4, 4])

    def test_classification_predict(self):
        """
            Test predict of an input sample using a classification NN
        """
        pass