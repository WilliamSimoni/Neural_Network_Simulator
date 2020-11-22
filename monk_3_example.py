"""
    Test our NN model using Monk3 dataset
"""
import numpy as np
from utility import read_monk_data
from neural_network import NeuralNetwork
from layer import OutputLayer, HiddenLayer
import weight_initializer as wi
import activation_function as af

def monk_example():
    """
        Test NN model using monk3 dataset
    """
    nn = NeuralNetwork(100, 'euclidean_loss', '', 0.8, 0.005, nn_type="batch", batch_size=1)

    #create three layers

    train_data, train_label, valid_data, valid_label = read_monk_data("dataset/monks-3.train", 0.8)

    layer1 = HiddenLayer(weights=wi.xavier_initializer(20, len(train_data[0])),
                         learning_rates=np.full((20, len(train_data[0]) + 1),  0.8),
                         activation=af.TanH())
    layer2 = OutputLayer(weights=wi.xavier_initializer(1, 20),
                         learning_rates=np.full((1, 21), 0.8),
                         activation=af.Sigmoid())

    nn.add_layer(layer1)
    nn.add_layer(layer2)

    training_examples = list(zip(train_data, train_label))
    validation_examples = list(zip(valid_data, valid_label))

    report = nn.fit(training_examples, validation_examples)
    report.plot_loss()

if __name__ == "__main__":
    monk_example()
