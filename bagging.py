import math
import numpy as np
from random import choice
from neural_network import NeuralNetwork
from report import Report


class Bagging():

    def __init__(self, sample_size, max_epochs_training=500, bootstrap=True):
        self.sample_size = sample_size
        self.models = []
        self.min_tr_errors = []
        self.bootstrap = bootstrap
        self.max_epochs_training = max_epochs_training

    def _generate_sample(self, dataset):
        """perform bootstrap with resampling

        Args:
            dataset (list): list over which perform the bootstrap

        Returns:
            list
        """
        return [choice(dataset) for i in range(0, self.sample_size)]

    def add_neural_network(self, model, min_error=1e-12):
        """add a neural network to the ensemble

        Args:
            model (NeuralNetwork): neural network to add to the ensemble
            min_error ([type], optional): min error for early stopping. Defaults to 1e-12.
        """
        self.models.append(model)
        self.min_tr_errors.append(min_error)

    def fit(self, training_set, validation_set=None, test_set=None):
        """perform training 

        Args:
            training_set (np.array): list used for training
            validation_set (np.array, optional): list used for validation. Defaults to None.

        Returns:
            Report: report that contains information about the training 
        """
        final_report = Report(self.max_epochs_training, 0)
        training_reports = []

        # training
        for i in range(0, len(self.models)):
            # if bootstrap is true then we perform _generate_sample(Bootstramp with resampling), otherwise we simply use
            # the original training set
            report = self.models[i].fit(self._generate_sample(training_set), validation_set, test_set, min_error=self.min_tr_errors[i]
                                        ) if self.bootstrap else self.models[i].fit(training_set, validation_set, test_set, min_error=self.min_tr_errors[i])
            
            #print("model ", i, ": ", report.get_vl_accuracy())
            
            training_reports.append(report)

        # calculate the mean of every report
        final_report.training_error = np.mean(
            [report.training_error for report in training_reports], axis=0)
        final_report.validation_error = np.mean(
            [report.validation_error for report in training_reports], axis=0)
        final_report.validation_error = np.mean(
            [report.test_error for report in training_reports], axis=0)
        final_report.training_accuracy = np.mean(
            [report.training_accuracy for report in training_reports], axis=0)
        final_report.validation_accuracy = np.mean(
            [report.validation_accuracy for report in training_reports], axis=0)
        final_report.test_accuracy = np.mean(
            [report.test_accuracy for report in training_reports], axis=0)

        #print(final_report.training_accuracy[0])
        return final_report

    def predict(self, sample):
        """
            Predict method implement the predict operation to make prediction
            about predicted output of a sample

            Parameters:
                sample: represents the feature space of an sample

            Return: the predicted target over the sample
        """
        return np.mean([model.predict(sample) for model in self.models], axis=0)
