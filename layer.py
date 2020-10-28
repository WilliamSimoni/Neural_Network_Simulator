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

        self.net = 0

        self.errors = np.empty([self.num_unit])

        self.inputs = 0

        self.old_delta_w = np.zeros(weights.shape)

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
    
    def get_errors(self):
        return self.errors

    def get_weights(self):
        return self.weights

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

        self.inputs = input_values

        self.net = np.array([np.inner(unit_weights, input_values) for unit_weights in self.weights])

        return np.array(self.activation.output(self.net))

    def update_weight(self, momentum_rate = 0, regularization_rate = 0):
        new_delta_w = np.multiply(self.learning_rates, np.outer(self.errors, self.inputs))
        self.weights += new_delta_w + self.old_delta_w * momentum_rate
        self.old_delta_w = new_delta_w

    def error_signal(self):
        pass

class OutputLayer(Layer):

    def __init__(self, weights, learning_rates, activation):
        super().__init__(weights, learning_rates, activation)
    
    def error_signal(self, target, output):
        self.errors = self.activation.derivative(self.net) * (target - output)

class HiddenLayer(Layer):

    def __init__(self, weights, learning_rates, activation):
        super().__init__(weights, learning_rates, activation)

    def error_signal(self, downStreamErrors, downStreamWeights):
        self.errors = self.activation.derivative(self.net) * np.matmul(downStreamErrors, downStreamWeights[0:,1:])