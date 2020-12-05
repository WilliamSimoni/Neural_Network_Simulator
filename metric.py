"""
    Metric functions to evaluate models
"""
import numpy as np

def classification_accuracy(predicted, targets):
    """
        Accuracy function for Classification
        Params:
            predicted (list): list of predicted values of ML Model
            targets (list): list of target values of ML Model
        Return a value between 0.0 and 1.0 and more is near 1.0 more
        the ML model provide right prediction
    """
    predicted = [0.1 if predict < 0.5 else 0.9 for predict in predicted]
    
    correct_prediction = 0
    for (predict, target) in zip(predicted, targets):
        if predict == target:
            correct_prediction += 1
    return correct_prediction / len(predicted)

def euclidean_loss(predicted, targets):
    """
        Calculate the Euclidean Loss for Regression
        Param:
            predict: predicted output layer
            target: a target value (array of values)
    """
    #if not isinstance(predicted, np.ndarray) or not isinstance(predicted, array):
    #    raise ValueError('predict must be a np.ndarray object or an array')
    #if not isinstance(targets, np.ndarray) or not isinstance(targets, array):
    #    raise ValueError('Target must be a np.ndarray object or an array ')
    #len = len(predicted)

    return np.mean([np.linalg.norm(predict - target) for predict, target in list(zip(predicted, targets))])

metric_functions = {
    'classification_accuracy': classification_accuracy,
    'euclidean_loss': euclidean_loss,
}
