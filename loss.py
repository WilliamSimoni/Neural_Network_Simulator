"""
    Loss module used to compute the Loss function of ML model
"""
import numpy as np

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

