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
    return np.linalg.norm(predict - target)

