import numpy as np
import copy
from utility import normalize_data, read_monk_data, read_cup_data
from neural_network import NeuralNetwork
from layer import OutputLayer, HiddenLayer
import weight_initializer as wi
import activation_function as af
import math
import cProfile
import learning_rate as lr

def split(dataset, num_subsets):
    min_num_element_per_subset = math.floor(len(dataset) / num_subsets)
    residual = len(dataset) - min_num_element_per_subset*num_subsets

    return [(k*min_num_element_per_subset + min(k, residual), (
        k+1) * min_num_element_per_subset + min(k+1, residual)) for k in range(0, num_subsets)]


def cross_validation(model, dataset, num_subsets):
    """cross validation implementation

    Args:
        model (NeuralNetwork): neural network from each fold iteration start
        dataset (array of tuple): data for training
        num_subsets (int): number of folds

    Returns:
        (float64, float64, float64, array of Report): 
            * mean validation error
            * standard deviation over the validation error
            * the mean training error when the validation error was minimum
            * list of all reports
    """
    #output of the crossvalidation
    sum_tr_err_with_best_vl_err = 0
    sum_error = 0
    errors = np.zeros(num_subsets)
    reports = []

    #get the indexes to break down the data set into the different folds
    splitted_data_set_index = split(dataset, num_subsets)
    print(splitted_data_set_index)

    for k in range(0, num_subsets):
        #create a deep copy of the model passed as argument
        model_k = copy.deepcopy(model)

        #dividing training and validation set
        training_set = dataset[:splitted_data_set_index[k]
                               [0]] + dataset[splitted_data_set_index[k][1]:]
        validation_set = dataset[splitted_data_set_index[k]
                                 [0]:splitted_data_set_index[k][1]]

        #traing the model
        report = model_k.fit(training_set, validation_set)
        print("Finished for k = {}".format(k))

        #update things for the cross validation result

        #get what was the training error when we reach the minimum validation error 
        sum_tr_err_with_best_vl_err += report.get_tr_err_with_best_vl_err()

        #add the error to the vector erros for calculating (at the end) the standard deviation
        errors[k] = report.get_vl_error()

        #sum the error to calculate the mean error (at the end)
        sum_error += errors[k]
        
        reports.append(report)

        # report.plot_loss()

    return sum_error/num_subsets, np.std(errors), sum_tr_err_with_best_vl_err/num_subsets, reports


nn = NeuralNetwork(500, 'euclidean_loss', '', 0.8,
                   0, nn_type="minibatch", batch_size=1)

# create three layers

train_data, train_label, _, _ = read_cup_data("dataset/ML-CUP20-TR.csv", 1)
#train_data, train_label, _, _ = read_monk_data("dataset/monks-1.train", 1)
train_data, train_label = normalize_data(train_data, train_label)

layer1 = HiddenLayer(weights=wi.xavier_initializer(30, len(train_data[0])),
                     learning_rates=lr.Constant(30, len(train_data[0]),  0.5),
                     activation=af.TanH())
layer2 = OutputLayer(weights=wi.xavier_initializer(2, 30),
                     learning_rates=lr.Constant(2, 30, 0.5),
                     activation=af.Linear())

nn.add_layer(layer1)
nn.add_layer(layer2)

training_examples = list(zip(train_data, train_label))
# print(training_examples)
cross_validation_res = cross_validation(nn, training_examples, 3)
# print(split(training_examples, 5))
#print(cross_validation_res)
#cProfile.run('cross_validation(nn, training_examples, 3)')
