"""
    Module grid_search to do the grid search for ML models
"""
from utility import normalize_data, read_cup_data
from cross_validation import cross_validation
import multiprocessing
import weight_initializer as wi
import activation_function as af

#Parameters which we conduct our GridSearch on our NN model
parameters = {
    'learning_rates': [0.01, 0.05, 0.1],
    'regularization': [0, 0.005, 0.01],
    'momentum': [0, 0.4, 0.8],
    'weight_initialization': [wi.xavier_initializer, wi.he_initializer],
    'activation_hidden': [af.TanH, af.Relu],
    'type_nn': ['batch'],
    'batch_size': [1],
    'topology': [(10, 10), (15, ), (20, 20)],
    'loss': 'mean_squared_error',
    'accuracy': 'euclidean_loss',
    'num_epoch': 500,
}

train_data, train_label, _, _ = read_cup_data("dataset/ML-CUP20-TR.csv")
train_data, train_label = normalize_data(train_data, train_label)
dataset = list(zip(train_data, train_label))

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
        self.loss = parameters['loss_function']
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
        models_param = self.initialize_models_param()

        for model_param in models_param:
            model = self.initialize_model(model_param)
            self.pool.apply_async(func=run,
                                  args=(model, self.results, model_param))

        self.pool.close()
        self.pool.join()

        #Sort results according to the accuracy of models
        self.results.sort(key=lambda x: x['accuracy_average_vl'], reverse=True)

        #Write to file results obtained
        self.write_results(save_path)

        return self.results[0]

    def initialize_models_param(self):
        """
            Initialize list of models parameters
            
            Return a shuffle list of models parameters to use to initialize a NN object
            with also layer information
        """

    def initialize_model(self, model_param):
        """
            Create NN model to use to execute a cross validation on it

            Param: 
                model_param(dict): dictionary of param to use to create NN object
            
            Return a NN model with also complete graph topology of the network
        """

    def write_results(self, file_path):
        """
            Write results obtained by GridSearch in a txt file
            Param:
                file_path(str): path where we will save our results on GridSearch
        """
        with open(file_path) as file:
            for item in self.results:
                file.write(
                    "{'accuracy_best_vl':" + str(item['accuracy_average_vl']) + ", 'learning_rate':" + 
                    str(item['model_param']['learning_rate']) + ", regularization:" + 
                    str(item['model_param']['regularization']) + ", momentum:" + 
                    str(item['model_param']['momentum']) + ", activation_hidden:" + 
                    str(item['model_param']['activation']) + ", weight_init: " + 
                    str(item['model_param']['weight_init']) + "topology:" + 
                    str(item['model_param']['topology']))
        return None

def run(model, results, model_param):
    """
        Proxy function where it will start cross validation on a configuration
        in an asyncro way

        Param:
            model(NeuralNetwork): NeuralNetwork object to use
            results(List): List of results obtained in GridSearch
            model_param(dict): dict of param of model object
        Return nothing but add result from cross validation and model_param in results list
    """       
    average, standard_deviation, average_best_vl, _ = cross_validation(model, dataset, 4)
    results.append({
        'accuracy_average_vl': average_best_vl,
        'model_param': model_param,
    })
     

