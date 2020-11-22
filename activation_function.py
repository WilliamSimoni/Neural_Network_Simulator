"""
    Activation Function module manages Activation function for ML models
"""
import numpy as np

class ActivationFunction():
    """
        Abstract class Activation Function used to represent
        an activation function
    """
    def output(self, x_val):
        """return the output of the function f(x_val)

        Args:
            x_val (numpy.ndarray): input for the activation function

        Returns:
            numpy.ndarray: output of the function
        """

    def derivative(self, x_val):
        """return f'(x_val) given x_val

        Args:
            x_val (numpy.ndarray): input for the derivative of the activation function f'(x_val)

        Returns:
            numpy.ndarray: output of the derivative
        """


class Linear(ActivationFunction):
    """Implementation of the linear function:

        properties:
            * range: (-oo,+oo)
            * 0-centered: YES
            * computation: easy

        graph:
                            |           x
                            |        x
                            |     x
                            |  x
            ----------------x----------------->
                         x  |
                      x     |
                   x        |
                x           |
    """

    def output(self, x_val):
        return x_val

    def derivative(self, x_val):
        return np.ones(x_val.shape)


class Sigmoid(ActivationFunction):
    """Implementation of the sigmoid function:

        properties:
            * range: (0,1)
            * 0-centered: NO
            * saturation: for negative and positive values
            * vanishing gradient: YES
            * computation: intensive

        graph:
                          1 |           x    x
                            |       x
                            |   x
                            |
                            x
                            |
                        x   |
                    x       |
            x---x------------------------------- 0
    """

    def output(self, x_val):
        return 1.0/(1.0 + np.exp(-x_val))

    def derivative_func(self, func):
        """return value of the derivative [f'(x)] given the function values on x [f(x)]

        Args:
            func (numpy.ndarray): array of function

        Returns:
            numpy.ndarray: return f'(x) given f(x)
        """
        return func * (1 - func)

    def derivative(self, x_val):
        return self.derivative_func(self.output(x_val))


class TanH(ActivationFunction):
    """Implementation of the tanh function

        properties:
            * range: (-1,1)
            * 0-centered: YES
            * saturation: for negative and positive values
            * vanishing gradient: YES
            * computation: intensive

        graph:
                          1 |           x    x
                            |       x
                            |   x
                            |
            ----------------x----------------->
                            |
                        x   |
                    x       |
            x   x           | -1

    """
    def output(self, x_val):
        return np.round(np.tanh(x_val), 6)

    def derivative_func(self, func):
        """return value of the derivative [f'(x)] given the function values on x [f(x)]

        Args:
            func (numpy.ndarray): array of function

        Returns:
            numpy.ndarray: return f'(x) given f(x)
        """
        return np.round(1 - np.square(func), 6)

    def derivative(self, x_val):
        return self.derivative_func(self.output(x_val))


class Relu(ActivationFunction):
    """Implementation of the relu function

        properties:
            * range: (0, +oo)
            * 0-centered: No
            * saturation: for negative values
            * vanishing gradient: YES (better then sigmoid and tanh)
            * computation: easy

        graph:
                            |           x
                            |        x
                            |     x
                            |  x
            x--x--x--x--x--x|------------------->
                            |
                            |
                            |
                            |

    """

    def output(self, x_val):
        return np.maximum(0, x_val)

    def derivative(self, x_val):
        return 1.0*(x_val > 0)


class LeakyRelu(ActivationFunction):
    """Implementation of the leaky relu function

        properties:
            * range: (-oo, +oo)
            * 0-centered: Close
            * saturation: NO
            * vanishing gradient: NO
            * computation: easy

        graph:
                            |           x
                            |        x
                            |     x
                            |  x
            ----------------x----------------->
                        x   |
                   x        |
               x            |
           x                |

    """

    def __init__(self, slope):
        self.slope = slope

    def output(self, x_val):
        return np.maximum(self.slope*x_val, x_val)

    def derivative(self, x_val):
        result = np.zeros_like(x_val)
        result[x_val < 0] = self.slope
        result[x_val >= 0] = 1
        return result


class SoftPlus(ActivationFunction):
    """Implementation of the soft plus function

        properties:
            * range: (0, +oo)
            * vanishing gradient: NO
            * computation: intensive

        graph:
                          1 |
                            |
                            |                       x
                            |                    x
                            |                 x
                            |              x
                            |           x
                            |      x
            x---x---x---x---x------------------- 0

    """

    def output(self, x_val):
        return np.log(1 + np.exp(x_val))

    def derivative(self, x_val):
        return 1 / (1 + np.exp(-x_val))
