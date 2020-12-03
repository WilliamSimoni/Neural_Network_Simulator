"""
Neural Network module implement a feedforward Neural Network
"""
import math
import json
import numpy as np
from layer import Layer
from neural_exception import InvalidNeuralNetwork
from report import Report
from loss import loss_functions
from metric import metric_functions
import tqdm


class NeuralNetwork:
    """
        Neural Network class to represent a feedforward Neural Network
    """

    def __init__(self, max_epochs, loss='euclidean_loss', metric='',
                 momentum_rate=0, regularization_rate=0, nn_type="SGD",
                 batch_size=1, type_classifier="classification"):
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
            loss (string): Indicate the loss function to use to evaluate the model
            metric(string): indicate the metric used to evaluate the model, like Accuracy
        """
        self.max_epochs = self.check_max_epochs(max_epochs)
        self.input_dimension = 0
        self.output_dimension = 0
        self.type = self.check_nn_type(nn_type)
        self.batch_size = self.check_batch_size(batch_size, )
        self.type_classifier = self.check_type_classifier(type_classifier)

        # note: this is not a np.ndarray object
        self.layers = []
        self.momentum_rate = self.check_momentum(momentum_rate)
        self.regularization_rate = self.check_regularization(
            regularization_rate)
        self.metric = self.check_metric(metric)
        self.loss = self.check_loss(loss)

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

    def check_loss(self, loss):
        """
            Check valid loss function inserted in NN constructor
            Param:
                loss(string): name of loss function to use to evaluate NN model
            Return:
                loss if is a valid loss function otherwise raise InvalidNeuralNetwork exception.
        """
        if loss in loss_functions:
            return loss
        else:
            raise InvalidNeuralNetwork()

    def check_max_epochs(self, max_epochs):
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

    def check_metric(self, metric):
        """
            Check metric value inserted in NN constructor
            Param:
                metric(string): name of metric function used to evaluate NN model
            Return:
                metric if is a valid metric function otherwise raise InvalidNeuralNetwork exception
        """

        if metric in metric_functions or metric == '':
            return metric
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

    def add_layer(self, layer):
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

        # the first layer added define the input dimension of the neural network
        if len(self.layers) == 0:
            self.input_dimension = layer.get_num_input()
        # the new layer must have an input dimension equal
        # to the number of units in the last layer added
        elif layer.get_num_input() != self.output_dimension:
            raise ValueError(
                "The number of input for this new layer must be equal to previous layer")

        # the last layer inserted define the output dimension
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
        # sample dimension controlled in _feedwardSignal
        return self._feedward_signal(sample)

    def _feedward_signal(self, sample):
        """
            FeedwardSignal feedward the signal from input to output of a feedforward NN

            Parameters:
                sample: represent the feature space of an sample

            Precondition:
                The length of sample is equal to input dimension in NN

            Return: the predicted output obtained after propagation of signal
        """
        if sample.shape[1] != self.input_dimension:
            raise ValueError

        input_layer = sample

        for layer in self.layers:
            output_layer = layer.function_signal(input_layer)
            input_layer = output_layer

        return output_layer

    def fit(self, training_examples, validation_samples=None, test_samples=None, min_error=1e-12):
        """[summary]

        Parameters:
            training_examples (array(tupla(input, target))): [description]
            validation_samples (array(tupla(input, target))): Validation samples (default None)
            test_samples (array(tupla(input, target))): Test samples to use in test (default None)
            min_training_error (): [description]
        """
        # create empty Report object
        report = Report(self.max_epochs, min_error)
        total_samples = len(training_examples)

        if self.type == "batch":
            self.batch_size = total_samples

        # executed epochs
        num_epochs = 0
        error = np.Inf
        num_window = math.ceil(total_samples // self.batch_size)

        inputs_training = np.array([elem[0] for elem in training_examples])
        targets_training = np.array([elem[1] for elem in training_examples])

        if validation_samples:
            inputs_validation = np.array([elem[0]
                                          for elem in validation_samples])
            targets_validation = np.array([elem[1]
                                           for elem in validation_samples])

        # ratio between batch size and the total number of samples
        batch_total_samples_ratio = self.batch_size/total_samples

        ex = training_examples[0]
        # stop when we execure max_epochs epochs or TODO training error

        for num_epochs in tqdm.tqdm(range(self.max_epochs), desc="fit"):

            # shuffle training examples
            np.random.shuffle(training_examples)

            # training

            for index in range(0, num_window):
                window_examples = training_examples[index * self.batch_size:
                                                    (index+1) * self.batch_size]

                # Backpropagate training examples
                self._back_propagation(
                    window_examples, batch_total_samples_ratio)

            training_predicted = self.predict(inputs_training)

            # calculate Training error
            error = loss_functions[self.loss](
                training_predicted,
                targets_training,
            ) 

            if self.metric != '':
                accuracy = metric_functions[self.metric](
                    training_predicted,
                    targets_training)
                report.add_training_accuracy(accuracy, num_epochs)

            report.add_training_error(error, num_epochs)

            if validation_samples:
                val_predicted = self.predict(inputs_validation)
                validation_error = loss_functions[self.loss](
                    val_predicted,
                    targets_validation,
                )
                if self.metric != '':
                    accuracy = metric_functions[self.metric](
                        val_predicted,
                        targets_validation)
                    report.add_validation_accuracy(accuracy, num_epochs)

                report.add_validation_error(
                    error, validation_error, num_epochs)

            if test_samples:
                test_error = loss_functions[self.loss](
                    [self.predict(test_example[0])
                     for test_example in test_samples],
                    [test_example[1] for test_example in test_samples],
                ) / len(test_samples)
                report.add_test_error(test_error, num_epochs)

            #print("Error during epoch {} is {}".format(num_epochs, error))
            # print("Predicted value during epoch {} is {}"
            #      .format(num_epochs, self.predict(ex[0])))
            # print("Target value during epoch {} is {}".format(
            #    num_epochs, ex[1]))
            #print("Num Epoch: ", num_epochs)

            # check error
            if error <= min_error:
                break

            # update the learning rate
            [layer.update_learning_rate(num_epochs) for layer in self.layers]

            # increase number of epochs
            num_epochs += 1

        return report

    def to_json(self):
        """
            Serialize NN object to a Json object
        """
        json_str = json.dumps(self.__dict__)
        return json_str

    def _back_propagation(self, samples, batch_total_samples_ratio):
        """execute a step of the backpropagation algorithm

        Parameters:
            samples (np.array): list of samples
            batch_total_samples_ratio (float): batch_size / len(samples)
        """
        """
        #Extended code using normal For
        for sample in samples:

            # calculate error signal (delta) of output units
            self.layers[-1].error_signal(sample[1], self.predict(sample[0]))
            self.layers[-1].update_delta_w()

           
            
            for index in range(len(self.layers[:-1]) - 1, -1, -1):
                self.layers[index].error_signal(self.layers[index+1].get_errors(),
                                                self.layers[index+1].get_weights())
                self.layers[index].update_delta_w()
        """
        # calculate error signal (delta) of output units
        targets = np.array([elem[1] for elem in samples])
        inputs = np.array([elem[0] for elem in samples])
        self.layers[-1].error_signal(targets, self.predict(inputs))

        # calculate error signal (delta) of hidden units
        [self.layers[index].error_signal(self.layers[index+1].get_errors(),
                                         self.layers[index+1].get_weights())
         for index in range(len(self.layers[:-1]) - 1, -1, -1)]

        # updating the weights in the neural network
        [layer.update_weight(
            self.batch_size, batch_total_samples_ratio,
            self.momentum_rate, self.regularization_rate)
         for layer in self.layers]

        """
        for layer in self.layers:
            layer.update_weight(
                self.batch_size, batch_total_samples_ratio,
                self.momentum_rate, self.regularization_rate)
        """
