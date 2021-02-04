"""
    Optimizer module implements several optimization algorithms for training NN models
"""
import numpy as np

class Optimizer:
    """
        General Optimizer strategy to training NN models
    """
    def _back_propagation(self, neural_network, samples, batch_total_samples_ratio,
                          loss_function):
        """
            Execute a step of the backpropagation algorithm
                Parameters:
                    neural_network(NeuralNetwork): the NN model to modify with backpropagation
                    samples (np.array): list of samples
                    batch_total_samples_ratio (float): batch_size / len(samples)
                    loss_function: Loss object used to compute Loss derivative
        """
        # calculate error signal (delta) of output units
        targets = np.array([elem[1] for elem in samples])
        inputs = np.array([elem[0] for elem in samples])
        neural_network.layers[-1].error_signal(targets, neural_network.predict(inputs),
                                               loss=loss_function)

        # calculate error signal (delta) of hidden units
        [neural_network.layers[index].error_signal(neural_network.layers[index+1].get_errors(),
                                         neural_network.layers[index+1].get_weights())
        for index in range(len(neural_network.layers[:-1]) - 1, -1, -1)]

        # updating the weights in the neural network
        [self.update_weights(layer, neural_network.batch_size, 
                            batch_total_samples_ratio,
                            neural_network.regularization_rate,
                            neural_network.momentum_rate,
                           )
            for layer in neural_network.layers]
#        [layer.update_weight(
 #           self.batch_size, batch_total_samples_ratio,
  #          self.momentum_rate, self.regularization_rate)
   #      for layer in neural_network.layers]

    def update_weights(self):
        """
            Update weights of a NN model
        """
        pass

class GradientDescent(Optimizer):
    """
        Gradient Descent Optimizer
    """
    def update_weights(self, layer, batch_size, batch_total_samples_ratio,
                       regularization, momentum):
        """
            Update weights using the Gradient Descent Approach

            Param:
                layer: Layer object used to update their weights
                batch_size: size of batch used
                batch_total_samples_ratio: ratio batch_size / len(dataset)
                regularization: regularization rate to use during weight's update
                momentum: momentum rate to use during weight's update

                    Formula:
            The j-th weight of the i-th unit is update as follow:

                W[i][j] = W[i][j] + learning_rate[i][j] * errors[i] * inputs[j] +
                          momentum_rate * old_delta_w +
                          reularization_rate * W[i][j]
        
        """
            # calculating the new delta
            # new_delta_w[i][j] = learning_rate[i][j] * errors[i] * inputs[j]
        new_delta_w = np.multiply(
            layer.learning_rates.value() / batch_size, 
            layer.current_delta_w
        )

        # regularization (no for bias)
        layer.weights[0:, 1:] -= batch_total_samples_ratio * \
                        regularization * layer.weights[0:, 1:]

        # adding delta_w and momentum
        layer.weights += new_delta_w + layer.old_delta_w * momentum

        # updating old_delta_w for the next update of the weights
        layer.old_delta_w = new_delta_w

        

class AdaGrad(Optimizer):
    """
        AdaGrad Optimizer
    """


class Adam(Optimizer):
    """
        Adam Optimizer
    """

optimizer_implemented = {
    'Adam': Adam,
    'SGD': GradientDescent,
    'AdaGrad': AdaGrad,
}