"""
    Layer Module used to represent a layer of a NN
"""
import numpy as np
from activation_function import ActivationFunction
from learning_rate import LearningRate

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

            activation (ActivationFunction): each unit of this layer use this function
                                                as activation function
        """

        # checking parameters -------------------
        if not isinstance(weights, np.ndarray):
            raise ValueError('weights must be a np.ndarray object')
        if not isinstance(learning_rates, LearningRate):
            raise ValueError('learning_rates must be a np.ndarray object')
        if not isinstance(activation, ActivationFunction):
            raise ValueError('activation must be an activation function')
        
        if weights.shape != learning_rates.value().shape:
            raise ValueError(
                'weights and learning_rates must have the same shape')
        # ---------------------------------------

        self.weights = weights
        self.learning_rates = learning_rates
        self.activation = activation

        # num_unit = number of weights'/learning_rates' rows
        # num_input = number of weights'/learning_rates' columns
        self.num_unit, self.num_input = weights.shape

        # removing 1, because in weights there is also the bias column
        self.num_input -= 1

        self.net = 0

        # delta calculated in the last error signal execution
        self.errors = np.empty([self.num_unit])

        # contains the last input the layer has processed
        self.inputs = 0

        self.old_delta_w = np.zeros(weights.shape)
        self.current_delta_w = np.zeros(weights.shape)

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
        """To get the array of errors obtained once you have executed the error signal

        Returns:
            np.array: an array of floating-point. In particular,
                the i-th element of the returned array is the error
                of the i-th unit in the layer
        """
        return self.errors

    def get_weights(self):
        """To get the weights of each unit of the level.
        Returns:
            np.array: a matrix W of dimension self.get_num_unit * ( self.get_num_input + 1).
                W[i][j] is the j-th weight of the i-th unit.
        """
        return self.weights

    def update_learning_rate(self, epoch):
        self.learning_rates.update(epoch)

    def function_signal(self, input_values):
        """
            Calculate the propagated values of a layer using an activation function

            Parameters:
                input_values: values used as input in a Layer (it is the output of predicing layer)

            Return: output values of Layer units
        """
        # checking that the input is the right dimension
        if len(input_values) != self.num_input:
            raise ValueError

        # adding bias input to input_values
        input_values = np.concatenate(([1], input_values), axis=0)

        # updating the value of inputs
        self.inputs = input_values

        # calculating the value of the net. The value calculated is an array
        # whose i-th element is the net value of the i-th unit.
        self.net = np.matmul(self.weights, input_values)

        # returnig the value obtained applying the activation function
        # of the layer to the new nets result.
        return np.array(self.activation.output(self.net))

    def update_weight(self, batch_size, batch_total_samples_ratio,
                            momentum_rate=0, regularization_rate=0):
        """update the weights of the layers

        Parameters:
            batch_size (int): size of a batch used to compute average over batch
            batch_total_samples_ratio (float): batch_size / len(samples)
            momentum_rate (int, optional): the momentum rate used to update the weights.
                Defaults to 0.
            regularization_rate (int, optional): the regularization rate used to update the weights.
                                                    Defaults to 0.

        Formula:
            The j-th weight of the i-th unit is update as follow:

                W[i][j] = W[i][j] + learning_rate[i][j] * errors[i] * inputs[j] +
                          momentum_rate * old_delta_w +
                          reularization_rate * W[i][j]
        """
        # calculating the new delta
        # new_delta_w[i][j] = learning_rate[i][j] * errors[i] * inputs[j]
        new_delta_w = np.multiply(
            self.learning_rates.value() / batch_size, self.current_delta_w)

        # updating the weights

        # regularization (no for bias)
        self.weights[0:, 1:] -= batch_total_samples_ratio * \
            regularization_rate * self.weights[0:, 1:]

        # adding delta_w and momentum
        self.weights += new_delta_w + self.old_delta_w * momentum_rate

        # updating old_delta_w for the next update of the weights
        self.old_delta_w = new_delta_w

        # reinitialize things
        self.current_delta_w = np.zeros(self.weights.shape)
        self.errors = np.empty([self.num_unit])

    def error_signal(self, target, output):
        """abstract class

            implementation in output layer and input layer
        """


class OutputLayer(Layer):
    """
        Represent an Output Layer in NN model
        It is a subclass of Layer object
    """
    def __init__(self, weights, learning_rates, activation):
        """This function initialize an instance of the layer class

            Parameters:
                weights (numpy.ndarray): matrix, of num_unit * num_input + 1 elements,
                that contain the weights of the units in the layer (including the biases)

                learning_rates (numpy.ndarray): matrix, of unitNumber * inputNumber elements,
                    that contain the learning rates of the units in the layer (including the biases)

                activation (ActivationFunction): each unit of this layer use this function
                                                    as activation function
        """
        super().__init__(weights, learning_rates, activation)

    def error_signal(self, target, output):
        """implement the calculation of the error signal for an output layer

        Parameters:
            target (np.array): target for a specific pattern x
            output (np.array): the output of the layer for a specific pattern x

        Formula:
            for each unit i

                errors[i] = f'(net[i]) * (target[i] - output[i])
        """
        self.errors = np.round(
            (self.activation.derivative(self.net) * (target - output)), 8)
        self.current_delta_w += np.outer(self.errors, self.inputs)


class HiddenLayer(Layer):
    """
        Represent an Hidden Layer in our NN model and it is a subclass of Layer object
    """

    def __init__(self, weights, learning_rates, activation):
        """This function initialize an instance of the layer class

            Parameters:
                weights (numpy.ndarray): matrix, of num_unit * num_input + 1 elements,
                that contain the weights of the units in the layer (including the biases)

                learning_rates (numpy.ndarray): matrix, of unitNumber * inputNumber elements,
                that contain the learning rates of the units in the layer (including the biases)

                activation (ActivationFunction): each unit of this layer use this
                                                    function as activation function
        """
        super().__init__(weights, learning_rates, activation)

    def error_signal(self, downStreamErrors, downStreamWeights):
        """implement the calculation of the error signal for an hidden layer

        Parameters:
            downStreamErrors (np.array): error signals of the layer above
            downStreamWeights (np.array): weights of the layer above

        Formula:
            for each unit i, assuming the layer above has k units:

                errors[i] = f'(net[i]) * (downStreamWeights[0][i] * downStreamErrors[0] + ... +
                                                    downStreamWeights[k][i] * downStreamErrors[k])
        """
        self.errors = np.round((self.activation.derivative(self.net) *
                                np.matmul(downStreamErrors, downStreamWeights[0:, 1:])), 8)
        self.current_delta_w += np.outer(self.errors, self.inputs)
