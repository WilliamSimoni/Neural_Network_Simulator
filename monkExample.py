import utility
import numpy as np
from neuralNetwork import NeuralNetwork
from layer import OutputLayer, HiddenLayer
import weightInitializer as wi
import activationFunction as af

def monk_example():
    NN = NeuralNetwork(50, nn_type="minibatch", batch_size=1)

    #create three layers

    train_data, train_label, valid_data, valid_label = utility.read_monk_data("dataset/monks-1.train", 0.8)

    layer1 = HiddenLayer(wi.xavier_initializer(2,len(train_data[0])), np.full((2,len(train_data[0])+1), 0.1), af.Relu())
    layer2 = OutputLayer(wi.xavier_initializer(1, 2), np.full((1,3), 0.1), af.Sigmoid())

    NN.addLayer(layer1)
    NN.addLayer(layer2)

    report = NN.fit([(x, y) for x in train_data for y in train_label])
    report.plotLoss()
    report2 = NN.fit([(x, y) for x in valid_data for y in valid_label])
    report2.plotLoss()

if __name__ == "__main__":
    monk_example()