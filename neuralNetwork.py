"""
Neural Network module implement a feedforward Neural Network
"""
import numpy as np
from layer import Layer

class NeuralNetwork:
    """
        Neural Network class to represent a feedforward Neural Network
    """

    def predict(self, sample):
        """
            Predict method implement the predict operation to make prediction
            about predicted output of a sample

            Parameters:
                sample: represents the feature space of an sample

            Precondition:
                The length of sample is equal to input dimension in NN

            Return: the predicted target over the sample
        """
        if len(sample) != self.input_dimension:
            raise ValueError
        return self._feedwardSignal(sample)

    def _feedwardSignal(self, sample):
        """
            FeedwardSignal feedward the signal from input to output of a feedforward NN

            Parameters:
                sample: represent the feature space of an sample

            Precondition:
                The length of sample is equal to input dimension in NN
              
            Return: the predicted output obtained after propagation of signal
        """
        if len(sample) != self.input_dimension:
            raise ValueError
   
        input_layer = sample

        for layer in self.layers:
            output_layer = layer.function_signal(input_layer)
            input_layer = output_layer

        return output_layer
