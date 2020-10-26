"""
    Layer Module used to represent a layer of a NN
"""
import numpy as np
from activationFunction import Activation_function

class Layer:
    """
        Layer class represent a layer in a NN
    """

    def __init__(self, weights, learning_rates, activation):
        """This function initialize an instance of the layer class

        Parameters:
            weights (numpy.ndarray): matrix, of num_unit * num_input + 1 elements, 
            that contain the weights of the units in the layer (including the biases)

            learning_rates (numpy.ndarray): matrix, of unitNumber * inputNumber elements, 
            that contain the learning rates of the units in the layer (including the biases)

            activation (ActivationFunction): each unit of this layer use this function as activation function
        """

        # checking parameters -------------------
        if not isinstance(weights, np.ndarray):
            raise ValueError('weights must be a np.ndarray object')
        if not isinstance(learning_rates, np.ndarray):
            raise ValueError('learning_rates must be a np.ndarray object')
        if not isinstance(activation, Activation_function):
            raise ValueError('activation must be an activation function')

        if weights.shape != learning_rates.shape:
            raise ValueError(
                'weights and learning_rates must have the same shape')
        # ---------------------------------------

        self.weights = weights
        self.learning_rates = learning_rates
        self.activation = activation

        # num_unit = number of weights'/learning_rates' rows
        # num_input = number of weights'/learning_rates' columns
        self.num_unit, self.num_input = weights.shape
        
        #removing 1, because in weights there is also the bias column
        self.num_input -= 1

    def get_num_unit(self):
        """To get the number of unit in the layer

        Returns:
            int: the number of units in the layer
        """
        return self.num_unit

    def get_num_input(self):
        """To get the number of input for the layer

        Returns:
            int: the number of input for the layer (included the bias input)
        """
        return self.num_input
        
    def function_signal(self, input_values):
        """
            Calculate the propagated values of a layer using an activation function

            Parameters:
                input_values: values used as input in a Layer (it is the output of predicing layer)

            Return: output values of Layer units
        """
        if len(input_values) != self.num_input:
            raise ValueError
        
        #add bias input to input_values
        input_values = np.concatenate(([1], input_values), axis=0)

        return np.array([self.activation.output(np.inner(unit_weights, input_values))
                         for unit_weights in self.weights])
