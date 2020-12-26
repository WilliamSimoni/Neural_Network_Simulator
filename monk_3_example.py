"""
    Test our NN model using Monk3 dataset
"""
import numpy as np
import sys
sys.path.append('')
print(sys.path)
from utility import read_monk_data
from neural_network import NeuralNetwork
from layer import OutputLayer, HiddenLayer
import weight_initializer as wi
import activation_function as af
import learning_rate as lr
from bagging import Bagging 

def monk_example():
    """
        Test NN model using monk3 dataset
    """
    train_data, train_label, _, _ = read_monk_data("dataset/monks-3.train")
    test_data, test_label, _, _ = read_monk_data("dataset/monks-3.test")

    #create a bagger to perform bagging (bootstrap set to false)
    bagger = Bagging(len(train_data), 500, False)

    #create and add neural network to the bagger
    for _ in range(0,10):
        nn = NeuralNetwork(500, 'mean_squared_error', 'classification_accuracy', 0.8, 0.01, nn_type="batch", batch_size=1)
        
        layer1 = HiddenLayer(weights=wi.he_initializer(15, len(train_data[0])),
                            learning_rates=lr.Constant(15, len(train_data[0]), 0.8),
                            activation=af.Relu())
        layer2 = OutputLayer(weights=wi.he_initializer(1, 15),
                            learning_rates=lr.Constant(1, 15, 0.8),
                            activation=af.Sigmoid())
        
        nn.add_layer(layer1)
        nn.add_layer(layer2)

        bagger.add_neural_network(nn)
        

    training_examples = list(zip(train_data, train_label))
    test_examples = list(zip(test_data, test_label))

    #training
    report = bagger.fit(training_examples, test_examples)

    #report result
    print("training mse", report.training_error[-1])
    print("validation mse", report.get_vl_error())
    print("training accuracy", report.training_accuracy[-1])
    print("validation accuracy", report.get_vl_accuracy())
    report.plot_loss()
    report.plot_accuracy()

if __name__ == "__main__":
    monk_example()
