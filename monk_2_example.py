"""
    Test our NN model using Monk3 dataset
"""
import numpy as np
from utility import read_monk_data
from neural_network import NeuralNetwork
from layer import OutputLayer, HiddenLayer
import weight_initializer as wi
import activation_function as af
import learning_rate as lr


def monk_example():
    """
        Test NN model using monk3 dataset
    """
    nn = NeuralNetwork(400, 'mean_squared_error', 'classification_accuracy',
                       0.8, nn_type="batch", batch_size=1)

    #load data

    train_data, train_label, _, _ = read_monk_data("dataset/monks-2.train")
    test_data, test_label, _, _ = read_monk_data("dataset/monks-2.test")

    #create two layers
    
    layer1 = HiddenLayer(weights=wi.he_initializer(15, len(train_data[0])),
                         learning_rates=lr.Constant(15, len(train_data[0]), 0.7),
                         activation=af.Relu())
    layer2 = OutputLayer(weights=wi.he_initializer(1, 15),
                         learning_rates=lr.Constant(1, 15, 0.7),
                         activation=af.Sigmoid())

    nn.add_layer(layer1)
    nn.add_layer(layer2)

    #training 
    
    training_examples = list(zip(train_data, train_label))
    test_examples = list(zip(test_data, test_label))

    report = nn.fit(training_examples, test_examples)
    report.plot_loss()
    report.plot_accuracy()


if __name__ == "__main__":
    monk_example()
