"""
    Layer Module used to represent a layer of a NN
"""
import numpy as np

class Layer:
    """
        Layer class represent a layer in a NN
    """

    def function_signal(self, input_values):
        """
            Calculate the propagated values of a layer using an activation function

            Parameters:
                input_values: values used as input in a Layer (it is the output of predicing layer)

            Return: output values of Layer units
        """
        if len(input_values) != len(self.num_input):
            raise ValueError
    
        return np.array([self.activation.output(np.inner(unit_weights, input_values))
                         for unit_weights in self.weights])
