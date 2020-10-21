# Neural Network Simulator
A neural network deployed by Marco Natali and William Simoni for the Machine Learning course at the University of Pisa

The deployed neural network will be a **feedforward** network whose units in a layer will be connected to all the units belonging to the layer directly above and directly below.

## Class diagram

The neural network simulator consists of two classes:

* the **Neural Network** class apply forward and backpropagation on its layers
* the **Layer** class represent a layer of units of the neural network
* the **Activation function** abstract class represents an activation function

![Class diagram](https://user-images.githubusercontent.com/56754601/96553750-e678f780-12b5-11eb-9c03-63e387e19ac8.png)

### Neural Network Class

#### instance variables

the instance variables are:
* **layers**: ordered sequence of layer objects. 
* **momentum**: momentum function used for weights updatw
* **learningRateUpdate**: function that update the learning rates
* **momentumRate**: hyperparameter of momemtum rate
* **hiddenLayerNumber**: number of hidden layer
* **unitsPerHiddenlayer**: units per hidden layer
* **outputUnit**: number of output units
* **maxEpochs**: maximum number of epochs
* **regularizationRate**: regularization rate
* **regularizationFunction**: function used for regularization
* **batchSize**: size of the batch (e.g.: 1 is online)
* **inputDimension**: dimension of the input

#### methods

##### fit (public)
The fit method trains the neural network over a set of training examples.

input:
* trainingExamples: array of tuplas (x, target)

*pre-condition*: trainingExamples.length > 0 and x.dimension = inputDimension and target.dimension = outputUnit

*post-condition*: update the weights

##### predict (public)
Predict the target value over a sample

input:
* sample: feature space X that represents a feature space of an example

*pre-condition*: sample.dimension = inputDimension

*return*: returns the predicted target over the sample

##### feedForwardSignal (private)
Propagate the signal from input to output layer

input:
* sample: feature space X that represents a feature space of an example

*pre-condition*: sample.dimension = inputDimension

*return*: returns the predicted target over the sample

##### backPropagation (private)
Backpropagate the error from output to input layer

input: 
* exampleSample: 
* targetPredicted:


*post-condition*: update the weights

### Layer Class

A layer that contains **h** units with **i** inputs, stores the units' weight in a matrix ![equation](https://latex.codecogs.com/gif.latex?W%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bh*i%7D), the units' bias in a vector *b* ![equation](https://latex.codecogs.com/gif.latex?%5Cin%20%5Cmathbb%7BR%7D%5E%7Bh%7D), and the weights' learning rate in a matrix ![equation](https://latex.codecogs.com/gif.latex?E%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bh*i%7D). Each unit of a layer has the same activation function defined when an instance of the class is created.

There are two specialized classes of the layer class:
* the **Output layer class** that represents an output layer
* the **Hidden layer class** that represents an hidden layer

The two classes differ only on the error signal method. 

#### methods

##### functionSignal 

return the function signal

#### errorSignal

return the error signal

##### updateWeight

update weights 

## Function signal

Suppose to have a neural network with **i** inputs, **H** hidden layers with **h** units, and an output layer with **o** units. Therefore:
* each hidden layer has a matrix ![equation](https://latex.codecogs.com/gif.latex?W_%7Bl%7D%5Cin%20%5Cmathbb%7BR%7D%5E%7Bh*h%7D) and a vector ![equation](https://latex.codecogs.com/gif.latex?b_%7Bl%7D%5Cin%20%5Cmathbb%7BR%7D%5E%7Bh%7D), where *l* is the level of the layer; e.g. the first layer has l = 1 a the last layer has l = H. 
* The output layer has a matrix ![equation](https://latex.codecogs.com/gif.latex?W_%7Bo%7D%5Cin%20%5Cmathbb%7BR%7D%5E%7Bo*o%7D) and a vector ![equation](https://latex.codecogs.com/gif.latex?b_%7Bo%7D%5Cin%20%5Cmathbb%7BR%7D%5E%7Bo%7D).

The feedback signal for the first level is calculated as follow: TODO

# TODO (28/10/2020)

* function signal in layer (Marco)
* feedforwardSignal in neural network (Marco)
* predict in neural network (Marco)
* layer and NN constructor (William)
* activation function (William)


