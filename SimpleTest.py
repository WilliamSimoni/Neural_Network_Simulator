import numpy as np
from neuralNetwork import NeuralNetwork
from layer import OutputLayer, HiddenLayer
import weightInitializer as wi
import activationFunction as af

def simple_test():
    NN = NeuralNetwork(1000, 0.1, 0.001, nn_type="minibatch", batch_size=1, type_classifier="regression")

    #create three layers

    layer1 = HiddenLayer(wi.xavier_initializer(4,2), np.full((4,3), 0.01), af.Relu())
    layer2 = OutputLayer(wi.xavier_initializer(1,4), np.full((1,5), 0.01), af.TanH())

    training_examples = []

    for i in range(0,30):
        sample = (np.random.randint(0,10,size=2), np.array(0.9))
        training_examples.append(sample)

    for i in range(0,30):
        sample = (np.random.randint(-11,-1,size=2), np.array(-0.9))
        training_examples.append(sample)

    training_examples = np.array(training_examples, dtype=object)

    test = []

    for i in range(0,10):
        sample = (np.random.randint(0,10,size=2), np.array(0.9))
        test.append(sample)

    for i in range(0,10):
        sample = (np.random.randint(-11,-1,size=2), np.array(-0.9))
        test.append(sample)

    test = np.array(test, dtype=object)

    np.random.shuffle(test)

    #add layers to NN
    NN.addLayer(layer1)
    NN.addLayer(layer2)

    report = NN.fit(training_examples)

    report.plotLoss()

if __name__ == "__main__":
    simple_test()