import numpy as np
from neuralNetwork import NeuralNetwork
from layer import Layer, OutputLayer, HiddenLayer
import weightInitializer as wi
import activationFunction as af

NN = NeuralNetwork(19, 0.1)

#create three layers

layer1 = HiddenLayer(wi.xavier_initializer(3,5), np.full((5,4), 0.001), af.Linear())
layer2 = HiddenLayer(wi.xavier_initializer(5,7), np.full((7,6), 0.001), af.Linear())
layer3 = HiddenLayer(wi.xavier_initializer(7,2), np.full((2,8), 0.001), af.Linear())
layer4 = OutputLayer(wi.xavier_initializer(2,3), np.full((3,3), 0.001), af.Linear())



#add layers to NN
NN.addLayer(layer1)
NN.addLayer(layer2)
NN.addLayer(layer3)
NN.addLayer(layer4)

x = np.array([1, 2, 3])
target = np.array([1, 3, 4])

print(np.linalg.norm(target - NN.predict(x)))

for i in range(0,50):
    NN._back_propagation(x, target)

print(np.linalg.norm(target - NN.predict(x)))
