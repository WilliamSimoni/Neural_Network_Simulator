import numpy as np
import copy
from utility import normalize_data, read_monk_data, read_cup_data, denormalize_data
from neural_network import NeuralNetwork
from layer import OutputLayer, HiddenLayer
import weight_initializer as wi
import activation_function as af
import math
import cProfile
import learning_rate as lr
from metric import metric_functions

def split(dataset, num_subsets):
    """return the indices into which split the dataset into num_subsets subsets

    Args:
        dataset (list): dataset to be split
        num_subsets (int): number of subsets into which split the dataset

    Returns:
        list of tuple: indices into which split the dataset into num_subsets subsets. 
        For instance, with k = 2 and a dataset of 400 patterns, the method returns
        ((0,200),(200, 400)). 
    """
    min_num_element_per_subset = math.floor(len(dataset) / num_subsets)
    residual = len(dataset) - min_num_element_per_subset*num_subsets

    return [(k*min_num_element_per_subset + min(k, residual), (
        k+1) * min_num_element_per_subset + min(k+1, residual)) for k in range(0, num_subsets)]


def cross_validation(model, dataset, num_subsets, den_label=None):
    """cross validation implementation

    Args:
        model (NeuralNetwork): neural network from each fold iteration start
        dataset (array of tuple): data for training
        num_subsets (int): number of folds
        den_label ((float, float), optional): tupla of the form (mean, variance) used for denormalization.
        Defaults to None. If not indicated, cross-validation does not perform any 
        denormalization assuming that data is not normalized.  

    Returns:
        (float64, float64, float64, array of Report): 
            * mean validation error
            * standard deviation over the validation error
            * the mean training error when the validation error was minimum
            * list of all reports
    """
    # output of the crossvalidation
    sum_tr_err_with_best_vl_err = 0
    errors = np.zeros(num_subsets)
    reports = []

    # get the indexes to break down the data set into the different folds
    splitted_data_set_index = split(dataset, num_subsets)

    for k in range(0, num_subsets):
        # create a deep copy of the model passed as argument
        model_k = model.deepcopy()
        # dividing training and validation set
        training_set = dataset[:splitted_data_set_index[k]
                               [0]] + dataset[splitted_data_set_index[k][1]:]
        validation_set = dataset[splitted_data_set_index[k]
                                 [0]:splitted_data_set_index[k][1]]

        # traing the model
        report = model_k.fit(training_set, validation_set)
        print("Finished for k = {}".format(k))

        # update things for the cross validation result

        # get what was the training error when we reach the minimum validation error
        sum_tr_err_with_best_vl_err += report.get_tr_err_with_best_vl_err()

        inputs_validation = np.array([elem[0]
                                      for elem in validation_set])
        targets_validation = np.array([elem[1]
                                       for elem in validation_set])

        # add the error to the vector erros for calculating (at the end) the standard deviation and the mean accuracy
        error = 0

        #if den_label, then the result predicted by the hypothesis is denormalized 
        # as for the targets in the validation set
        if den_label:
            predicted_test_data = denormalize_data(
                model_k.predict(inputs_validation), den_label)
            error = metric_functions['euclidean_loss'](
                predicted_test_data,
                denormalize_data(targets_validation, den_label)
            )
        else:
            error = report.get_vl_accuracy()

        errors[k] = error

        reports.append(report)

        #to look at the accuracy plot
        # report.plot_accuracy()

    return np.round(np.mean(errors), 8), np.round(np.std(errors), 8), np.round(sum_tr_err_with_best_vl_err/num_subsets, 8), reports
