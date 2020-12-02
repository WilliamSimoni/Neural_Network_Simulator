"""
    Module grid_search to do the grid search for ML models
"""

class GridSearch():
    """
        GridSearch class to do grid_search
    """

    def __init__(self, parameters):
        """
            Initialize parameters which will be done Grid Search
        """
        self.learning_rates = parameters['learning_rates']
        self.regularization = parameters['regularization']
        self.momentum = parameters['momentum']
        self.weight_initialization = parameters['weight_initialization']
        self.activation_hidden = parameters['activation_hidden']
        self.type_nn = parameters['type_nn']
        self.batch_size = 1
        self.num_layer = parameters['num_layer']
        self.num_hidden_unit = parameters['num_hidden_unit']

    def 

