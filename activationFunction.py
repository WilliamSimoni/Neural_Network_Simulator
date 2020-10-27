import numpy as np

class Activation_function():

    def output(self, input):
        """return the output of the function f(input)

        Args:
            input (numpy.ndarray): input for the activation function

        Returns:
            numpy.ndarray: output of the function
        """

    def derivative(self, input):
        """return f'(input) given input

        Args:
            input (numpy.ndarray): input for the derivative of the activation function f'(input)

        Returns:
            numpy.ndarray: output of the derivative
        """


class Linear(Activation_function):
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

    def output(self, input):
        return input

    def derivative(self, input):
        return np.ones(input.shape)


class Sigmoid(Activation_function):
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

    def output(self, input):
        return 1.0/(1.0 + np.exp(-input))

    def derivativeF(self, function_value):
        """return value of the derivative [f'(x)] given the function values on x [f(x)]

        Args:
            f (numpy.ndarray): array of function

        Returns:
            numpy.ndarray: return f'(x) given f(x)
        """
        return function_value * (1 - function_value)

    def derivative(self, input):
        return self.derivativeF(self.output(input))


class TanH(Activation_function):
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

    def output(self, input):
        return np.tanh(x)

    def derivativeF(self, function_value):
        """return value of the derivative [f'(x)] given the function values on x [f(x)]

        Args:
            f (numpy.ndarray): array of function

        Returns:
            numpy.ndarray: return f'(x) given f(x)
        """
        return 1 - np.square(function_value)

    def derivative(self, input):
        return self.derivativeF(self.output(input))


class Relu(Activation_function):
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

    def output(self, input):
        return np.maximum(0,input)

    def derivative(self, input):
        return 1.0*(input > 0)


class LeakyRelu(Activation_function):
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

    def output(self, input):
        return np.maximum(self.slope*input, input)

    def derivative(self, input):
        result = np.zeros_like(input)
        result[input < 0] = self.slope
        result[input >= 0] = 1
        return result


class SoftPlus(Activation_function):
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

    def output(self, input):
        return np.log(1 + np.exp(input))

    def derivative(self, input):
        return 1 / (1 + np.exp(-input))


"""
# Example

# create activationFunction object
activationFunction = SoftPlus()

# define input
input = np.array([-3, 3])

#calculate the output
output = activationFunction.output(input)
print(output)

# calculate the derivative
derivative = activationFunction.derivative(input)
print(derivative)
"""