"""
    Loss module used to compute the Loss function of ML model
"""
import numpy as np
import layer

def euclidean_loss(predict, target):
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
    return np.linalg.norm(predict - target)


def cross_entropy(predicted, targets):
    """
        Calculate the Cross Entropy Loss for Classification
        Param:
            predicted: predicted output for all samples
            targets: target value provided by dataset for all samples
    """
    return -1 * np.sum([target * np.log(predict) + (1.0 - target) * np.log(1.0 - predict) 
                    for predict, target in list(zip(predicted, targets))])
