import numpy as np
from neuralNetwork import NeuralNetwork
from layer import OutputLayer, HiddenLayer
import weightInitializer as wi
import activationFunction as af

NN = NeuralNetwork(1000, 0.1, nn_type="minibatch", batch_size=1)

#create three layers

layer1 = HiddenLayer(wi.xavier_initializer(4,2), np.full((4,3), 0.01), af.Relu())
layer2 = OutputLayer(wi.xavier_initializer(1,4), np.full((1,5), 0.01), af.Linear())

training_examples = []

for i in range(0,30):
    sample = (np.random.randint(0,10,size=2), np.array(1))
    training_examples.append(sample)

for i in range(0,30):
    sample = (np.random.randint(-11,-1,size=2), np.array(-1))
    training_examples.append(sample)

training_examples = np.array(training_examples, dtype=object)

test = []

for i in range(0,10):
    sample = (np.random.randint(0,10,size=2), np.array(1))
    test.append(sample)

for i in range(0,10):
    sample = (np.random.randint(-11,-1,size=2), np.array(-1))
    test.append(sample)

test = np.array(test, dtype=object)

np.random.shuffle(test)

#add layers to NN
NN.addLayer(layer1)
NN.addLayer(layer2)

NN.fit(training_examples)

#test
for t in test:
    print(NN.predict(t[0]), t[0], t[1])