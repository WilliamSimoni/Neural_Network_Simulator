"""
Neural Network module implement a feedforward Neural Network
"""
import numpy as np
from layer import Layer, OutputLayer, HiddenLayer

class NeuralNetwork:
    """
        Neural Network class to represent a feedforward Neural Network
    """

    def __init__(self, max_epochs, momentum_rate = 0):
        """create an instance of neural network class

        Args:
            momentum_rate (int, optional): momentum_rate used for learning. Defaults to 0.
        """
        # checking parameters -------------------
        # TODO
        # ---------------------------------------

        #TODO DELETE MAX_EPOCHS?

        self.max_epochs = max_epochs
        self.input_dimension = 0
        self.output_dimension = 0

        # note: this is not a np.ndarray object
        self.layers = []
        self.momentum_rate = momentum_rate

    def addLayer(self, layer):

        """ add a layer in the neural network

            Parameters:
                layer (Layer): layer to be added. The layer must have a number
                of input equal to the unit of the previous layer

            Raises:
                ValueError: the layer is not a Layer object
                ValueError: The number of input for this new layer is not equal
                  to the number of unit of the previous layer in the neural network

            Example:
                this is a neural network with two layers

                      o   o   o
                    o   o   o   o

                Then we execute neuralNetwork.addLayer(layer) 
                where layer has 2 units with 3 inputs(o o):

                        o   o
                      o   o   o
                    o   o   o   o

        """

        if not isinstance(layer, Layer):
            raise ValueError('layer must be a Layer object')
        
        #the first layer added define the input dimension of the neural network
        if len(self.layers) == 0:
            self.input_dimension = layer.get_num_input()
        #the new layer must have an input dimension equal to the number of unit of the last layer added
        elif layer.get_num_input() != self.output_dimension:
            raise ValueError(
                "The number of input for this new layer must be equal to previous layer")

        #the last layer inserted define the output dimension
        self.output_dimension = layer.get_num_unit()

        self.layers.append(layer)

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
    
    def _back_propagation(self, target_sample, target_predicted):
        """execute a step of the backpropagation algorithm

        Parameters:
            target_sample (np.array): the target for the sample x
            target_predicted (np.array): the output returned by the neural network for the sample x
        """
        # calculate error signal (delta) of output units
        self.layers[-1].error_signal(target_sample, target_predicted)
        
        #calculate error signal (delta) of hidden units
        #TODO problem if there is only one layer
        for index in range(len(self.layers[:-2]),-1,-1): 
            self.layers[index].error_signal(self.layers[index+1].get_errors(),self.layers[index+1].get_weights())
        
        #updatinf the weights in the neural network
        for layer in self.layers:
            layer.update_weight(self.momentum_rate)

        
            
