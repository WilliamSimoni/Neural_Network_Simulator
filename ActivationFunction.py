import numpy as np


class ActivationFunction():

    def output(self, x):
        """return the output of the function f(x)

        Args:
            x (numpy.ndarray): input for the activation function

        Returns:
            numpy.ndarray: output of the function
        """

    def derivative(self, x):
        """return f'(x) given x

        Args:
            x (numpy.ndarray): input for the derivative of the activation function f'(x)

        Returns:
            numpy.ndarray: output of the derivative
        """


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

    def output(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def derivativeF(self, f):
        """return value of the derivative [f'(x)] given the function values on x [f(x)]

        Args:
            f (numpy.ndarray): array of function

        Returns:
            numpy.ndarray: return f'(x) given f(x)
        """
        return f * (1 - f)

    def derivative(self, x):
        return self.derivativeF(self.output(x))


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

    def output(self, x):
        return np.tanh(x)

    def derivativeF(self, f):
        """return value of the derivative [f'(x)] given the function values on x [f(x)]

        Args:
            f (numpy.ndarray): array of function

        Returns:
            numpy.ndarray: return f'(x) given f(x)
        """
        return (1 - np.square(f))

    def derivative(self, x):
        return self.derivativeF(self.output(x))


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

    def output(self, x):
        return 1.0*(x > 0)

    def derivative(self, x):
        return 1.0*(x > 0)


class LeakyRelu(ActivationFunction):
    """Implementation of the leaky relu function

        properties:
            * range: (-oo, +oo)
            * 0-centered: Close
            * saturation: NO
            * vanishing gradient: NO
            * computation: easy

        graph:
                            |          x
                            |       x
                            |    x
                            | x
            ----------------x----------------->
                        x   |
                    x       |
                x           |
            x               |

    """

    def __init__(self, slope):
        self.slope = slope

    def output(self, x):
        return np.maximum(self.slope*x,x)

    def derivative(self, x):
        result = np.zeros_like(x)
        result[x < 0] = self.slope
        result[x >= 0] = 1
        return result

"""
# Example

# create activationFunction object
activationFunction = Sigmoid()

# define x
x = np.array([5.54, -25.43, 342134])

# calculate the derivative
output = activationFunction.output(x)

gradient = activationFunction.derivative(x)

print(gradient)
"""