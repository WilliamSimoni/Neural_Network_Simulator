import numpy as np
import copy
from utility import read_monk_data
from neural_network import NeuralNetwork
from layer import OutputLayer, HiddenLayer
import weight_initializer as wi
import activation_function as af
import math


def split(dataset, num_subsets):
    min_num_element_per_subset = math.floor(len(dataset) / num_subsets)
    residual = len(dataset) - min_num_element_per_subset*num_subsets

    return [(k*min_num_element_per_subset + min(k, residual), (
        k+1) * min_num_element_per_subset + min(k+1, residual)) for k in range(0, num_subsets)]


def cross_validation(model, dataset, num_subsets):
    splitted_data_set_index = split(dataset, num_subsets)

    for k in range(0, num_subsets):
        model_k = copy.deepcopy(model)
        training_set = dataset[:splitted_data_set_index[k][0]] + dataset[splitted_data_set_index[k][1]:]
        validation_set = dataset[splitted_data_set_index[k][0]:splitted_data_set_index[k][1]]
        report=model_k.fit(training_set, validation_set)
        print("Finished for k = {}".format(k))
        report.plot_loss()


nn = NeuralNetwork(200, 'euclidean_loss', '', 0.8,
                   0.005, nn_type="batch", batch_size=1)

# create three layers

train_data, train_label, _, _ = read_monk_data("dataset/monks-3.train", 1)

layer1 = HiddenLayer(weights=wi.xavier_initializer(15, len(train_data[0])),
                     learning_rates=np.full(
    (15, len(train_data[0]) + 1),  0.8),
    activation=af.TanH())
layer2 = OutputLayer(weights=wi.xavier_initializer(1, 15),
                     learning_rates=np.full((1, 16), 0.8),
                     activation=af.Sigmoid())

nn.add_layer(layer1)
nn.add_layer(layer2)

training_examples = list(zip(train_data[:19], train_label[:16]))
# print(training_examples)
cross_validation(nn, training_examples, 5)
# print(split(training_examples, 5))
