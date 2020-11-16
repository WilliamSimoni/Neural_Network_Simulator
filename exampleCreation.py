import numpy as np
from neuralNetwork import NeuralNetwork
from layer import OutputLayer, HiddenLayer
import weightInitializer as wi
import activationFunction as af

NN = NeuralNetwork(10000, nn_type="minibatch", batch_size=1)

#create three layers

wHidden = np.array([[0, 0.84886225, -0.70838177], 
              [0, 0.88296627, -0.0148878], 
              [0, -0.32618086, 0.8117482]])

wOut = np.array([[0, 0.65348894, 0.7163801, 0.12492884]])

layer1 = HiddenLayer(wHidden, np.full((3,3), 0.1), af.Sigmoid())
layer2 = OutputLayer(wOut, np.full((1,4), 0.1), af.Sigmoid())



#add layers to NN
NN.addLayer(layer1)
NN.addLayer(layer2)

x = np.array([([1,1],[0]), ([0,1], [1]), ([1,0], [1]), ([0,0], [0])])

report = NN.fit(x)

report.plotLoss()