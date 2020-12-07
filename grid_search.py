"""
    Module grid_search to do the grid search for ML models
"""
from cross_validation import cross_validation
import multiprocessing
import itertools

import weight_initializer as wi
import activation_function as af

parameters = {
    'learning_rates': [0.01, 0.05, 0.1],
    'regularization': [0, 0.005, 0.01],
    'momentum': [0, 0.4, 0.8],
    'weight_initialization': [wi.xavier_initializer, wi.he_initializer],
    'activation_hidden': [af.TanH, af.Relu],
    'type_nn': ['batch'],
    'batch_size': [1],
    'topology': [(10, 10), (15, ), (20, 20)],
    'loss': ['mean_squared_error'],
    'accuracy': ['euclidean_loss'],
    'num_epoch': [500],
}


class GridSearch():
    """
        GridSearch class to do grid_search
    """

    def __init__(self, parameters, n_threads=5):
        """
            Initialize parameters which will be done Grid Search
        """
        self.learning_rates = parameters['learning_rates']
        self.regularization = parameters['regularization']
        self.momentum = parameters['momentum']
        self.weight_initialization = parameters['weight_initialization']
        self.activation_hidden = parameters['activation_hidden']
        self.type_nn = parameters['type_nn']
        self.batch_size = parameters['batch_size']
        self.topology = parameters['topology']
        self.num_epoch = parameters['num_epoch']
        self.loss = parameters['loss']
        self.accuracy = parameters['accuracy']
        self.pool = multiprocessing.Pool(multiprocessing.cpu_count()) if n_threads is None else \
            multiprocessing.Pool(n_threads)
        self.results = multiprocessing.Manager().list()

    def grid_search(self, save_path='grid_results/grid'):
        """
            Execute Grid Search

            Param: 
                save_path(str): string of file path 

        """
        params = [
            self.learning_rates,
            self.regularization,
            self.momentum,
            self.weight_initialization,
            self.activation_hidden,
            self.type_nn,
            self.batch_size,
            self.topology,
            self.num_epoch,
            self.loss,
            self.accuracy,
        ]

        for model_param in itertools.product(*params):
            model = self.initialize_model(model_param)
            self.pool.apply_async(func=run,
                                  args=(model, self.results, model_param))

        self.pool.close()
        self.pool.join()

        # Sort results according to the accuracy of models

        # Write to file results obtained

        return best_model

    def initialize_model(self, model_param):
        """
            Create NN model to use to execute a cross validation on it

            Param: 
                model_param(dict): dictionary of param to use to create NN object

            Return a NN model with also complete graph topology of the network
        """
        print(model_param)

    def run(self):
        result = cross_validation()
        self.results.append(result, model_param)

    def write_results(self):
        pass

gs = GridSearch(parameters)
gs.grid_search()