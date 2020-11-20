import utility
import numpy as np
from neuralNetwork import NeuralNetwork
from layer import OutputLayer, HiddenLayer
import weightInitializer as wi
import activationFunction as af

def monk_example():
    NN = NeuralNetwork(100, 0.8, 0.005, nn_type="batch", batch_size=1)

    #create three layers

    train_data, train_label, valid_data, valid_label = utility.read_monk_data("dataset/monks-3.train", 0.8)

    
    layer1 = HiddenLayer(weights=wi.xavier_initializer(20, len(train_data[0])),
                         learning_rates=np.full((20, len(train_data[0]) + 1),  0.8),
                         activation=af.TanH())
    layer2 = OutputLayer(weights=wi.xavier_initializer(1, 20),
                         learning_rates=np.full((1,21), 0.8),
                         activation=af.Sigmoid())
    
    
    NN.addLayer(layer1)
    NN.addLayer(layer2)

    lenExample = len(train_data)

    training_examples = list(zip(train_data, train_label))

    validation_examples = list(zip(valid_data, valid_label))


    report = NN.fit(training_examples)
    report.plotLoss()
    report2 = NN.fit(validation_examples)
    report2.plotLoss()

if __name__ == "__main__":
    monk_example()