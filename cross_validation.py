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

train_data, train_label, test_data, test_label = read_cup_data("dataset/ML-CUP20-TR.csv", 0.8)
_, _, den_data, den_label = normalize_data(train_data, train_label)

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
    errors = np.zeros(num_subsets)
    reports = []

    #get the indexes to break down the data set into the different folds
    splitted_data_set_index = split(dataset, num_subsets)

    for k in range(0, num_subsets):
        #create a deep copy of the model passed as argument
        model_k = model.deepcopy()
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

        inputs_validation = np.array([elem[0]
                                          for elem in validation_set])
        targets_validation = np.array([elem[1]
                                           for elem in validation_set])
        
        #add the error to the vector erros for calculating (at the end) the standard deviation
        predicted_test_data = denormalize_data(model_k.predict(inputs_validation), den_label)
        error = metric_functions['euclidean_loss'](
                predicted_test_data,
                denormalize_data(targets_validation, den_label)
            )
        
        errors[k] = error

        print(error)

        reports.append(report)

        #report.plot_accuracy()

    return np.round(np.mean(errors), 8), np.round(np.std(errors), 8), np.round(sum_tr_err_with_best_vl_err/num_subsets, 8), reports

"""
nn = NeuralNetwork(500, 'mean_squared_error', 'euclidean_loss', 0.8,
                   0.01, nn_type="batch", batch_size=1)

# create three layers

train_data, train_label, _, _ = read_cup_data("dataset/ML-CUP20-TR.csv", 0.8)
#train_data, train_label, _, _ = read_monk_data("dataset/monks-1.train", 1)
train_data, train_label = normalize_data(train_data, train_label)

layer1 = HiddenLayer(weights=wi.xavier_initializer(20, len(train_data[0])),
                     learning_rates=lr.Constant(20, len(train_data[0]),  0.1),
                     activation=af.TanH())

layer2 = HiddenLayer(weights=wi.xavier_initializer(50, 50),
                     learning_rates=lr.Constant(50, 50,  0.2),
                     activation=af.TanH())
layer3 = OutputLayer(weights=wi.xavier_initializer(2, 20),
                     learning_rates=lr.Constant(2, 20, 0.1),
                     activation=af.Linear())

nn.add_layer(layer1)
#nn.add_layer(layer2)
nn.add_layer(layer3)

training_examples = list(zip(train_data, train_label))
# print(training_examples)
cross_validation_res = cross_validation(nn, training_examples, 3)
# print(split(training_examples, 5))
print(cross_validation_res)
#cProfile.run('cross_validation(nn, training_examples, 3)')
"""