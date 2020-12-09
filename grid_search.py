"""
    Module grid_search to do the grid search for ML models
"""
from layer import HiddenLayer, OutputLayer
from utility import normalize_data, read_cup_data
import cross_validation as cv
import multiprocessing
import itertools
import learning_rate as lr
import weight_initializer as wi
import activation_function as af
from neural_network import NeuralNetwork

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
    'loss': ['mean_squared_error'],
    'accuracy': ['euclidean_loss'],
    'num_epoch': [500],
}

train_data, train_label, test_data, test_label = read_cup_data("dataset/ML-CUP20-TR.csv", 0.8)
train_data, train_label = normalize_data(train_data, train_label)
dataset = list(zip(train_data, train_label))

def grid_search(params, dataset, num_features, output_dim, n_threads=5, save_path='grid_results/grid.txt'):
    """
        Execute Grid Search

        Param: 
            save_path(str): string of file path 

    """
    params = [
        params['learning_rates'],
        params['regularization'],
        params['momentum'],
        params['weight_initialization'],
        params['activation_hidden'],
        params['type_nn'],
        params['batch_size'],
        params['topology'],
        params['loss'],
        params['accuracy'],
        params['num_epoch']
    ]
    pool = multiprocessing.Pool(multiprocessing.cpu_count()) if n_threads is None else \
           multiprocessing.Pool(n_threads)
    results = multiprocessing.Manager().list()


    for model_param in list(itertools.product(*params)):
        model = initialize_model(model_param, num_features, output_dim)
        print("Model:", model)
        pool.apply_async(func=run,
                         args=(model, results, model_param, dataset))

    pool.close()
    pool.join()

    #Sort results according to the accuracy of models
    l_results = list(results)
    l_results = l_results.sort(key=lambda x: x['accuracy_average_vl'], reverse=True)

    #Write to file results obtained
    write_results(l_results, save_path)

    return results[0]


def initialize_model(model_param, num_features, output_dim):
    """
        Create NN model to use to execute a cross validation on it

        Param: 
            model_param(dict): dictionary of param to use to create NN object

        Return a NN model with also complete graph topology of the network
    """
    print(model_param)
    learning_rate = float(model_param[0])
    momentum = float(model_param[1])
    regularization = float(model_param[2])
    weight_initialization = model_param[3]
    activation = model_param[4]
    type_nn = model_param[5]
    batch_size = int(model_param[6])
    topology = model_param[7]
    loss = model_param[8]
    accuracy = model_param[9]
    num_epochs = int(model_param[10])
    
    print("Type Num_epochs:", type(num_epochs))
    #Create NN object model
    model = NeuralNetwork(num_epochs, loss, accuracy, momentum,
                          regularization, type_nn, batch_size)
        
    last_dim = num_features
    #Add Layers 
    for num_nodes in topology[1:]:
        layer = HiddenLayer(weight_initialization(num_nodes, last_dim),
                            lr.Constant(num_nodes, last_dim, learning_rate),
                            activation())
        model.add_layer(layer)
        last_dim = num_nodes
    output_layer = OutputLayer(weight_initialization(output_dim, last_dim),
                               lr.Constant(output_dim, last_dim, learning_rate),
                               af.Linear())
    model.add_layer(output_layer)
    return model

def write_results(results, file_path):
    """
        Write results obtained by GridSearch in a txt file
        Param:
            file_path(str): path where we will save our results on GridSearch
    """
    with open(file_path, 'w') as result_file:
        for item in results:
            result_file.write(
                "{'accuracy_best_vl':" + str(item['accuracy_average_vl']) + ", 'learning_rate':" + 
                str(item['model_param'][0]) + ", regularization:" + str(item['model_param'][2]) + 
                ", momentum:" + str(item['model_param'][1]) + ", activation_hidden:" + 
                str(item['model_param'][4]) + ", weight_init: " + str(item['model_param'][3])
                 + "topology:" + str(item['model_param'][7]) + "\n")
    return None

def run(model, results, model_param, dataset):
    """
        Proxy function where it will start cross validation on a configuration
        in an asyncro way

        Param:
            model(NeuralNetwork): NeuralNetwork object to use
            results(List): List of results obtained in GridSearch
            model_param(dict): dict of param of model object
            Return nothing but add result from cross validation and model_param in results list
    """       
    average, standard_deviation, average_best_vl, reports = cv.cross_validation(model, dataset, 4)
    results.append({
        'accuracy_average_vl': average_best_vl,
        'model_param': model_param,
    })
    print("Finish {} cross-validation".format(len(results)))
    

    
grid_search(parameters, dataset, len(train_data[0]), len(train_label[0]))