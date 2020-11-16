"""
Neural Network module implement a feedforward Neural Network
"""
import math
import numpy as np
from layer import Layer, OutputLayer, HiddenLayer
from neural_exception import InvalidNeuralNetwork
from report import Report
from loss import euclidean_loss

class NeuralNetwork:
    """
        Neural Network class to represent a feedforward Neural Network
    """

    def __init__(self, max_epochs, momentum_rate=0, regularization_rate=0, nn_type="SGD", batch_size=1, type_classifier="classification"):
        """create an instance of neural network class

        Args:
            max_epochs (int): number of maximum epochs used in param fitting.
            momentum_rate (int, optional): momentum_rate used for learning. Defaults to 0.
            regularization_rate(int,optional): regularization_rate used for learning. Defaults to 0
            nn_type (string, optional): type of Neural Network used. Default "SGD"
                Only SGD(Stocasthic Gradient Descent), batch and minibatch has been implemented.
            batch_size (int, optional): size of batch used, Default set to 1.
            type_classifier (string, optional): estabilish the type of classification used
                            Accepted values are "Classification" and "Regression"
        """
        # checking parameters -------------------
        # TODO
        # ---------------------------------------

        self.max_epochs = self.check_max_epochs(max_epochs)
        self.input_dimension = 0
        self.output_dimension = 0
        self.type = self.check_nn_type(nn_type)
        self.batch_size = self.check_batch_size(batch_size, )
        self.type_classifier = self.check_type_classifier(type_classifier)

        # note: this is not a np.ndarray object
        self.layers = []
        self.momentum_rate = self.check_momentum(momentum_rate)
        self.regularization_rate = self.check_regularization(regularization_rate)
    
    def check_batch_size(self, batch_size):
        """
            Check batch_size value inserted in NN constructor
            Param:
                batch_size(float): rate used as batch_size and should be > 0
            Return:
                batch_size is 1 if self.type is SGD
                batch_size if self.type is != SGD and batch_size > 0
                otherwise raise InvalidNeuralNetwork exception
        """
        if self.type == "SGD" and batch_size == 1:
            return batch_size
        elif batch_size > 0:
            return batch_size
        else:
            raise InvalidNeuralNetwork()
    
    def check_max_epochs(self,max_epochs):
        """
            Check max_epochs value inserted in constructor
            Param:
                max_epochs(int): number of epochs used in NN training and need to be > 0
            Return:
                max_epochs if >0 otherwise raise InvalidNeuralNetwork exception
        """
        if max_epochs > 0:
            return max_epochs
        else:
            raise InvalidNeuralNetwork()

    def check_momentum(self, momentum_rate):
        """
            Check momentum_rate value inserted in Constructor

            Param:
                momentum_rate(float): rate used as momentum and should be >= 0
            Return:
                momentum_rate if is >= 0 otherwise raise InvalidNeuralNetwork exception
        """
        if momentum_rate >= 0:
            print("Hello")
            return momentum_rate
        else:
            raise InvalidNeuralNetwork()

    def check_nn_type(self, nn_type):
        """
            Check type of Neural Network implemented in NN constructor
            Param:
                nn_type(string): indicate the type of NN implemented
                    Accepted value are SGD, minibatch and batch
            Return:
                nn_type is an accepted value otherwise raise InvalidNeuralNetwork exception
        """
        if nn_type in ["SGD", "minibatch", "batch"]:
           return nn_type
        else:
            raise InvalidNeuralNetwork()

    def check_regularization(self, regularization_rate):
        """
            Check regularization_rate value inserted in NN costructor
            Param:
                regularization_rate(float): rate used as regularization and should be >= 0
            Return:
                regularization_rate if is >= 0 otherwise raise InvalidNeuralNetwork exception
        """
        if regularization_rate >= 0:
            return regularization_rate
        else:
            raise InvalidNeuralNetwork()

    def check_type_classifier(self, type_classifier):
        """
            Check type of Classifier for NN model
            Param:
                type_classifier(string): type of classifier and valid value 
                                         are classification and regression
            Return:
                type_classifier if is a valid value otherwise raise InvalidNeuralNetwork exception
        """
        if type_classifier in ["classification", "regression"]:
            return type_classifier
        else:
            raise InvalidNeuralNetwork()

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
        #create empty Report object
        report = Report(self.max_epochs)

        if self.type == "batch":
            self.batch_size = len(training_examples[0])
        
        #executed epochs
        num_epochs = 0
        error = np.Inf
        num_window = math.ceil(len(training_examples) // self.batch_size)
        Direxample = training_examples[0]
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
            error = np.sum([euclidean_loss(self.predict(example[0]), example[1])
                           for example in training_examples]) / len(training_examples)

            report.add_training_error(error, num_epochs)

            #print("Error during epoch {} is {}".format(num_epochs, error))
            print("Predicted value during epoch {} is {}"
                  .format(num_epochs, self.predict(Direxample[0])))
            print("Target value during epoch {} is {}".format(num_epochs, Direxample[1]))
            print("Num Epoch: ", num_epochs)
            #increase number of epochs
            num_epochs += 1
        
        return report

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
            layer.update_weight(self.batch_size, self.momentum_rate, self.regularization_rate)

