"""
    Module weightInitializer define how to initialize weights in
    our NN simulator
"""
import numpy as np


def xavier_initializer(num_unit, num_input):
    """returns weight matrix to use in a layer (included bias in the first column)

        The xavier method apply the following rule:
        - w[i][j] = random() * 1/sqrt(num_input)
        - bias = 0

    Args:
        num_unit (int): number of unit in the layer
        num_input (int): number of input of the layer (no bias included in the counting)

    Returns:
        numpy.ndarray: returns weight matrix to use in a layer (included bias in the first column)
    """
    bias_weights = np.zeros((num_unit, 1))
    input_weights = np.random.randn(num_unit, num_input) * np.sqrt(1/num_input)
    return np.concatenate((bias_weights, input_weights), axis=1)


def he_initializer(num_unit, num_input):
    """returns weight matrix to use in a layer (included bias in the first column)

        The he method apply the following rule:
        - w[i][j] = random() * sqrt(2/num_input)
        - bias = 0

    Args:
        num_unit (int): number of unit in the layer
        num_input (int): number of input of the layer (no bias included in the counting)

    Returns:
        numpy.ndarray: returns weight matrix to use in a layer (included bias in the first column)
    """
    bias_weights = np.zeros((num_unit, 1))
    input_weights = np.random.randn(num_unit, num_input) * np.sqrt(2/num_input)
    return np.concatenate((bias_weights, input_weights), axis=1)


def all_zero_initializer(num_unit, num_input):
    """
    WARNING: ONLY FOR TESTING, BAD TO USE

    returns weight matrix to use in a layer (included bias in the first column).
    All elements are initialized with zero

    Args:
        num_unit (int): number of unit in the layer
        num_input (int): number of input of the layer (no bias included in the counting)

    Returns:
        numpy.ndarray: returns weight matrix to use in a layer (included bias in the first column)
    """
    bias_weights = np.zeros((num_unit, 1))
    input_weights = np.zeros((num_unit, num_input))
    return np.concatenate((bias_weights, input_weights), axis=1)


def big_random_initializer(num_unit, num_input):
    """
    WARNING: ONLY FOR TESTING, BAD TO USE

    returns weight matrix to use in a layer (included bias in the first column).
    All elements are initialized with random()*10

    Args:
        num_unit (int): number of unit in the layer
        num_input (int): number of input of the layer (no bias included in the counting)

    Returns:
        numpy.ndarray: returns weight matrix to use in a layer (included bias in the first column)
    """
    bias_weights = np.zeros((num_unit, 1))
    input_weights = np.random.randn(num_unit, num_input) * 10
    return np.concatenate((bias_weights, input_weights), axis=1)


def ranged_uniform_initializer(num_unit, num_input, min=-0.5, max=0.5):
    """
    returns weight matrix to use in a layer (included bias in the first column).
    All elements are initialized at random and are smaller than max and larger than min

    Args:
        num_unit (int): number of unit in the layer
        num_input (int): number of input of the layer (no bias included in the counting)
        min (float, optional): each generated weight is larger than min. Defaults to -0.5.
        max (float, optional): each generated weight is smaller than min. Defaults to 0.5.

    Returns:
        numpy.ndarray: returns weight matrix to use in a layer (included bias in the first column)
    """
    bias_weights = np.random.uniform(low=min, high=max, size=(num_unit, 1))
    input_weights = np.random.uniform(low=min, high=max, size=(num_unit, num_input))
    return np.concatenate((bias_weights, input_weights), axis=1)