# Neural Network Simulator
A neural network deployed by Marco Natali and William Simoni for the Machine Learning course at the University of Pisa

The deployed neural network will be a **feedforward** network whose units in a layer will be connected to all the units belonging to the layer directly above and directly below.

## Installation
To execute our neural network simulation you need to install some pip packages defined in requirement file using this command:
```
pip install -r requirement.txt
```

## Example of usage
In this example, we will create a neural network to solve the monk 3 task. The entire code is available in monk_3_example.py. First thing first, we need to create an instance of a neural network. To do that we just write:
```
nn = NeuralNetwork(500, 'SGD', 'mean_squared_error', 'classification_accuracy', 0.8, 0.01, batch_size=1)
```
The parameter we need to provide to the constructor of the neural network are in order:
* **max_epochs**: maximum number of epochs the training algorithm will perform. In our example, the training algorithm will stop after 500 epochs. 
* **optimizer**: indicate the Optimizer object used to train the model. SGD is at the moment the only available optimizer, however, it can be extended to other optimizers such as ADAM.
* **loss**: the loss function used to train the model. The available loss are "mean_squared_error" and "cross_entropy".
* **metric**: the metric used to evaluate the model, like classification accuracy or the euclidean loss. 
* **momentum_rate**: momentum_rate used for learning. Defaults to 0.
* **regularization_rate**: regularization_rate used for learning. Defaults to 0.
* **batch_size**:  size of batch used, Default set to 1 (stochastic SGD). It can be every number greater than 1.

We have now to define the layers that compose the topology of the neural network. In our example, we will create an hidden layer of four neurons and an output layer of 1 neuron:
```
 layer1 = HiddenLayer(weights=wi.he_initializer(4, len(train_data[0])),
                            learning_rates=lr.Constant(4, len(train_data[0]), 0.8),
                            activation=af.Relu())
```
```
 layer2 = OutputLayer(weights=wi.he_initializer(1, 4),
                            learning_rates=lr.Constant(1, 4, 0.8),
                            activation=af.Sigmoid())
```
There are two types of layer, the hidden layer and the output layer. Every neural network should have an output layer in the end. Every layer has the following parameters:
* **weights**: matrix of num_unit * num_input + 1 elements, that contain the weights of the units in the layer (including the biases). To create the matrix we can use the functions defined in weight_initializer.py. In our example, we used the he_initializer. 
* **learning_rate**: matrix of unitNumber * inputNumber elements, that contain the learning rates of the units in the layer (including the biases). To create the matrix we can use the functions defined in learning_rate.py. In our example, we used the Constant learning rate. 
* **activation**: activation function of the units in the layer. The functions are implemented in activation_function.py.

Eventually, we add the layers to the nueral network:
```
nn.add_layer(layer1)
nn.add_layer(layer2)
```

### Training
We first load the data:
```
train_data, train_label, _, _ = read_monk_data("dataset/monks-3.train")
test_data, test_label, _, _ = read_monk_data("dataset/monks-3.test")
```
We zip the loaded data:
```
training_examples = list(zip(train_data, train_label))
test_examples = list(zip(test_data, test_label))
```
Finally we can train the model:
```
report = nn.fit(training_examples, test_examples)
```
The fit function returns a report object that we can then use to plot some charts:
```
report.plot_loss()
report.plot_accuracy()
```
