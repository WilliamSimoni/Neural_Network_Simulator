"""
Neural Network module implement a feedforward Neural Network
"""
import numpy as np
import math
from layer import Layer, OutputLayer, HiddenLayer
from neural_exception import InvalidNeuralNetwork

class NeuralNetwork:
    """
        Neural Network class to represent a feedforward Neural Network
    """

    def __init__(self, max_epochs, momentum_rate=0, nn_type="SGD", batch_size=1):
        """create an instance of neural network class

        Args:
            max_epochs (int): number of maximum epochs used in param fitting.
            momentum_rate (int, optional): momentum_rate used for learning. Defaults to 0.
            nn_type (string, optional): type of Neural Network used. Default "SGD"
                Only SGD(Stocasthic Gradient Descent), batch and minibatch has been implemented.
            batch_size (int, optional): size of batch used, Default set to 1.
        """
        # checking parameters -------------------
        # TODO
        # ---------------------------------------

        self.max_epochs = max_epochs
        self.input_dimension = 0
        self.output_dimension = 0
        self.type = nn_type
        self.batch_size = batch_size

        if nn_type == "SGD" and batch_size != 1:
            print("Invalid Batch Size. Reset to valid value 1")
            self.batch_size = 1
        if nn_type not in ("batch", "SGD", "minibatch"):
            raise InvalidNeuralNetwork

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
        #the new layer must have an input dimension equal
        # to the number of units in the last layer added
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
        #sample dimension controlled in _feedwardSignal

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

    def fit(self, training_examples, min_training_error=1e-12):
        """[summary]

        Parameters:
            training_examples (array(tupla(input, target))): [description]
            min_training_error (): [description]
        """

        #executed epochs
        num_epochs = 0
        error = np.Inf
        num_window = math.ceil(len(training_examples) // self.batch_size)

        #stop when we execure max_epochs epochs or TODO training error
        while(num_epochs < self.max_epochs or error <= min_training_error):

            #shuffle training examples
            np.random.shuffle(training_examples)

            #training

            for index in range(0, num_window):
                window_examples = training_examples[index * self.batch_size:
                                                    (index+1) * self.batch_size]
        
                #Backpropagate training examples
                for example in window_examples:
                    self._back_propagation([(example[1],
                                             self.predict(example[0]))
                                            for example in window_examples])

            #calculate euclidean error
            error = np.sum([np.linalg.norm(self.predict(example[0]) - example[1])
                           for example in training_examples]) / len(training_examples)

            print("Error during epoch {} is {}".format(num_epochs, error))
            print("Predicted value during epoch {} is {}"
                  .format(num_epochs, self.predict(training_examples[0][0])))
            print("Target value during epoch {} is {}".format(num_epochs, training_examples[0][1]))

            #increase number of epochs
            num_epochs += 1



    def _back_propagation(self, target_samples):
        """execute a step of the backpropagation algorithm

        Parameters:
            target_samples (np.array): list of (target, predicted_output) element

        """

        for target, predicted_target in target_samples:

            # calculate error signal (delta) of output units
            self.layers[-1].error_signal(target, predicted_target)

            #calculate error signal (delta) of hidden units
            for index in range(len(self.layers[:-1]) - 1, -1, -1):
                self.layers[index].error_signal(self.layers[index+1].get_errors(),
                                                self.layers[index+1].get_weights())

        # updating the weights in the neural network
        for layer in self.layers:
            layer.update_weight(self.batch_size, self.momentum_rate)

