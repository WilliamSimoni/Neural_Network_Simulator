# Neural Network Simulator
A neural network deployed by Marco Natali and William Simoni for the Machine Learning course at the University of Pisa

The deployed neural network will be a **feedforward** network whose units in a layer will be connected to all the units belonging to the layer directly above and directly below.

## Class diagram

The neural network simulator consists of two classes:

* the **Neural Network** class apply forward and backpropagation on its layers
* the **Layer** class represent a layer of units of the neural network

![Class diagram](https://user-images.githubusercontent.com/56754601/96553750-e678f780-12b5-11eb-9c03-63e387e19ac8.png)

### Layer Class

A layer that contains **h** units, stores the units' weight in a matrix *W* ![equation](https://latex.codecogs.com/gif.latex?%5Cin%20%5Cmathbb%7BR%7D%5E%7Bh*h%7D), the units' bias in a vector *b* ![equation](https://latex.codecogs.com/gif.latex?%5Cin%20%5Cmathbb%7BR%7D%5E%7Bh%7D), and the weights' learning rate in a matrix *E* ![equation](https://latex.codecogs.com/gif.latex?%5Cin%20%5Cmathbb%7BR%7D%5E%7Bh*h%7D). Each unit of a layer has the same activation function defined when an instance of the class is created.

## Feedback signal

Suppose to have a neural network with **i** inputs, **H** hidden layers with **h** units, and an output layer with **o** units. Therefore:
* each hidden layer has a matrix ![equation](https://latex.codecogs.com/gif.latex?W_%7Bl%7D%5Cin%20%5Cmathbb%7BR%7D%5E%7Bh*h%7D) and a vector ![equation](https://latex.codecogs.com/gif.latex?b_%7Bl%7D%5Cin%20%5Cmathbb%7BR%7D%5E%7Bh%7D), where *l* is the level of the layer; e.g. the first layer has l = 1 a the last layer has l = H. 
* The output layer has a matrix ![equation](https://latex.codecogs.com/gif.latex?W_%7Bo%7D%5Cin%20%5Cmathbb%7BR%7D%5E%7Bo*o%7D) and a vector ![equation](https://latex.codecogs.com/gif.latex?b_%7Bo%7D%5Cin%20%5Cmathbb%7BR%7D%5E%7Bo%7D).

The feedback signal for the first level is calculated as follow: TODO
