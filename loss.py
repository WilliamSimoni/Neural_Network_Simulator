"""
    Loss module used to compute the Loss function of ML model
"""
import numpy as np



def cross_entropy(predicted, targets):
    """
        Calculate the Cross Entropy Loss for Classification
        Param:
            predicted: predicted output for all samples
            targets: target value provided by dataset for all samples

        This implementation avoids the extreme values of the logatithm function by returning:
         0            if target==predict
        -log(1e-6)    if target==1-predict instead of -log(0) that would be -∞

    sum_w = 0
    for layer in layers:
        w = layer.get_weights()
        sum_w += np.sum(w*w)
    """
    eps = 1e-6
    return -1 * np.mean([target * np.log(max(predict, eps))
                        if target > 0.5 else np.log(max(1 - predict, eps))
                    for predict, target in list(zip(predicted, targets))])

def mean_squared_error(predicted, targets):
    """
        Calculate the Mean Squared Error used usually in regression task
        Param:
            predicted: predicted output for all samples
            targets: target value provided by dataset for all samples
    """
    return np.sum([(predict - target)**2 
                   for predict, target in list(zip(predicted, targets))]) / len(predicted)

loss_functions = {
    'cross_entropy': cross_entropy,
    'mean_squared_error': mean_squared_error,
}
