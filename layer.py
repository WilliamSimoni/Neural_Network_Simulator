"""
    Layer Module used to represent a layer of a NN
"""
import numpy as np
import activationFunction


class Layer:
    """
        Layer class represent a layer in a NN
    """

    def __init__(self, weights, learning_rates, num_unit, num_input, activation):
        """This function initialize an instance of the layer class

        Args:
            weights (numpy.ndarray): matrix, of num_unit * num_input + 1 elements, that contain the weights of the units in the layer
            learning_rates (numpy.ndarray): matrix, of unitNumber * inputNumber elements, that contain the learning rates associated with the units in the layer
            num_unit (int): number of hidden units that compose the layer
            num_unit (int): number of input for each unit of the layer
            activation (ActivationFunction): each unit of this layer use this function as activation function
        """

        # checking parameters -------------------
        # TODO
        # ---------------------------------------

        self.weights = weights
        self.learning_rates = learning_rates
        self.num_unit = num_unit

        # num_input + 1 because there are num_input inputs + the bias input
        self.num_input = num_input + 1

        self.activation = activation

    def function_signal(self, input_values):
        """
            Calculate the propagated values of a layer using an activation function

            Parameters:
                input_values: values used as input in a Layer (it is the output of predicing layer)

            Return: output values of Layer units
        """
        if len(input_values) != self.num_input:
            raise ValueError

        return np.array([self.activation.output(np.inner(unit_weights, input_values))
                         for unit_weights in self.weights])
