"""
    Metric functions to evaluate models
"""

def accuracy(predicted, targets):
    """
        Accuracy function
        Params:
            predicted (list): list of predicted values of ML Model
            targets (list): list of target values of ML Model
        Return a value between 0.0 and 1.0 and more is near 1.0 more
        the ML model provide right prediction
    """



metric_functions = {
    'accuracy': accuracy,
}