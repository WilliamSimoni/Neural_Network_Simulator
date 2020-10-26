import numpy as np
from neuralNetwork import NeuralNetwork
from layer import Layer
import weightInitializer as wi
import activationFunction as af

NN = NeuralNetwork(19)

#create three layers
layer1 = Layer(wi.xavier_initializer(3,5), wi.xavier_initializer(3,5), af.TanH())
layer2 = Layer(wi.xavier_initializer(5,7), wi.xavier_initializer(5,7), af.TanH())
layer3 = Layer(wi.xavier_initializer(7,1), wi.xavier_initializer(7,1), af.Linear())

#add layers to NN
NN.addLayer(layer1)
NN.addLayer(layer2)
NN.addLayer(layer3)

x = np.array([1, 2, 3])

print(NN.predict(x))