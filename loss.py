"""
    Loss module used to compute the Loss function of ML model
"""
import numpy as np

def euclidean_loss(predicted, targets):
    """
        Calculate the Euclidean Loss for Regression
        Param:
            predict: predicted output layer
            target: a target value (array of values)
    """
    #if not isinstance(predict, np.ndarray):
     #       raise ValueError('predict must be a np.ndarray object')
    #if not isinstance(target, np.ndarray):
    #        raise ValueError('Target must be a np.ndarray object')
    return np.sum([np.linalg.norm(predict - target)
                   for predict, target in list(zip(predicted, targets))])


def cross_entropy(predicted, targets):
    """
        Calculate the Cross Entropy Loss for Classification
        Param:
            predicted: predicted output for all samples
            targets: target value provided by dataset for all samples
    """
    #sum_w = 0
    #for layer in layers:
     #   w = layer.get_weights()
     #   sum_w += np.sum(w*w)
    return -1 * np.sum([target * np.log(predict) + (1.0 - target) * np.log(1.0 - predict)
                    for predict, target in list(zip(predicted, targets))])


loss_functions = {
    'euclidean_loss': euclidean_loss,
    'cross_entropy': cross_entropy,
}
