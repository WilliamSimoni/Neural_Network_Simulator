import numpy as np
from neuralNetwork import NeuralNetwork
from layer import OutputLayer, HiddenLayer
import weightInitializer as wi
import activationFunction as af

NN = NeuralNetwork(1000, 0.1, nn_type="minibatch", batch_size=2)

#create three layers

layer1 = HiddenLayer(wi.xavier_initializer(5,3), np.full((5,4), 0.01), af.Linear())
layer2 = HiddenLayer(wi.xavier_initializer(7,5), np.full((7,6), 0.01), af.Linear())
layer3 = HiddenLayer(wi.xavier_initializer(2,7), np.full((2,8), 0.01), af.Linear())
layer4 = OutputLayer(wi.xavier_initializer(3,2), np.full((3,3), 0.01), af.Linear())



#add layers to NN
NN.addLayer(layer1)
NN.addLayer(layer2)
NN.addLayer(layer3)
NN.addLayer(layer4)

x1 = np.array([1, 1, 1])
target1 = np.array([1, 0, 0])

x2 = np.array([2, 1, 1])
target2 = np.array([0, 1, 0])

x3 = np.array([3, 2, 1])
target3 = np.array([0, 0, 1])

x4 = np.array([3, 2, 1])
target4 = np.array([0, 0, 1])

training_examples = np.array([(x1,target1), (x2, target2), (x3, target3), (x4, target4)])

NN.fit(training_examples)



"""
print(np.linalg.norm(target - NN.predict(x)))

for i in range(0,50):
    NN._back_propagation(target, NN.predict(x))

print(np.linalg.norm(target - NN.predict(x)))
"""