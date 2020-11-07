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
        num_input (int): number of input of the layer (no bias included in the counting)
        num_unit (int): number of unit in the layer

    Returns:
        numpy.ndarray: returns weight matrix to use in a layer (included bias in the first column)
    """
    bias_weights = np.zeros((num_unit, 1))
    input_weights = np.random.normal(size=[num_unit, num_input]) * np.sqrt(1/num_input)
    return np.concatenate((bias_weights, input_weights), axis=1)
