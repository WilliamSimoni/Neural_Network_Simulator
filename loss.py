"""
    Loss module used to compute the Loss function of ML model
"""
import numpy as np

class Loss:
    """
        Abstract Class used to represent a Loss function
    """

    def loss(self, predicted, targets):
        """
            Compute the loss value
            Param:
                predicted: predicted output for all samples
                targets: target value provided by dataset for all samples
        """
        

    def derivative(self, predicted, targets):
        """
            Compute the Derivative of Loss function
            Param:
                predicted: predicted output for all samples
                targets: target value provided by dataset for all samples
        """
        

class CrossEntropy(Loss):
    """
        CrossEntropy Loss function
    """
    def loss(self, predicted, targets):
        """
            Calculate the Cross Entropy Loss for Classification
            Param:
                predicted: predicted output for all samples
                targets: target value provided by dataset for all samples

            This implementation avoids the extreme values of the logatithm function by returning:
            0            if target==predict
           -log(1e-6)    if target==1-predict instead of -log(0) that would be -infty

        sum_w = 0
        for layer in layers:
            w = layer.get_weights()
            sum_w += np.sum(w*w)
        """
        eps = 1e-6
        return -1 * np.mean([target * np.log(max(predict, eps))
                            if target > 0.5 else np.log(max(1 - predict, eps))
                            for predict, target in list(zip(predicted, targets))])

    def derivative(self, predicted, targets):
        """
            Calculate the derivative of Cross Entropy loss
            Param:
                predicted: predicted output for all samples
                targets: target value provided by dataset for all samples
        """

class MeanSquareError(Loss):
    """
        Mean Square Error Loss function
    """
    def loss(self, predicted, targets):
        """
            Calculate the Mean Square error for Regression task
            Param:
                predicted: predicted output for all samples
                targets: target value provided by dataset for all samples
        """
        return np.sum([(predict - target)**2
                   for predict, target in list(zip(predicted, targets))]) / len(predicted)

    def derivative(self, predict, target):
        """
            Calculate the derivative of Mean Square error loss using the k target
            used in Backpropagation
            Param:
                predict: k predicted output
                targets: k target value provided by dataset
        """
        return np.array([predict - target])


loss_functions = {
    'mean_squared_error': MeanSquareError(),
    'cross_entropy': CrossEntropy(),
}
