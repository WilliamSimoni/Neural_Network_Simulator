"""
    Module grid_search to do the grid search for ML models
"""
from optimizer import Optimizer
from layer import HiddenLayer, OutputLayer
from utility import normalize_data, read_cup_data, denormalize_data
import cross_validation as cv
import multiprocessing
import csv
import itertools
import time
import learning_rate as lr
import weight_initializer as wi
import activation_function as af
from neural_network import NeuralNetwork
from bagging import Bagging
from loss import loss_functions
from metric import metric_functions

# Parameters which we conduct our GridSearch on our NN model
parameters = {
    'learning_rates': [0.1, 0.13, 0.15],
    'regularization': [0.0005, 0.00075, 0.001, 0.0025, 0.005],
    'momentum': [0.6, 0.8, 1.0],
    'weight_initialization': [wi.xavier_initializer, wi.ranged_uniform_initializer],
    'activation_hidden': [af.TanH, af.Relu],
    'batch_size': [1],
    'topology': [(30, 20), (20, 20), (10, 5, 5), (15, 15), (20,)],
    'loss': ['mean_squared_error'],
    'accuracy': ['euclidean_loss'],
    'optimizer': 'SGD',
    'num_epoch': [500],
}

train_data, train_label, test_data, test_label = read_cup_data(
    "dataset/ML-CUP20-TR.csv", 0.8)
train_data, train_label, _, _ = normalize_data(train_data, train_label)
dataset = list(zip(train_data, train_label))


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
    average_vl, sd_vl, average_tr_error_best_vl, reports = cv.cross_validation(
        model, dataset, 4)
    results.append({
        'accuracy_average_vl': average_vl,
        'accuracy_sd_vl': sd_vl,
        'average_tr_error_best_vl': average_tr_error_best_vl,
        'model_param': model_param,
    })
    print("Finish {} cross-validation".format(len(results)))


def initialize_model(model_param, num_features, output_dim):
    """
        Create NN model to use to execute a cross validation on it

        Param:
            model_param(dict): dictionary of param to use to create NN object

        Return a NN model with also complete graph topology of the network
    """
    print(model_param)
    learning_rate = model_param[0]
    regularization = model_param[1]
    momentum = model_param[2]
    weight_initialization = model_param[3]
    activation = model_param[4]
    batch_size = model_param[5]
    topology = model_param[6]
    loss = model_param[7]
    accuracy = model_param[8]
    optimizer = model_param[9]
    num_epochs = model_param[10]

    # Create NN object model
    model = NeuralNetwork(num_epochs, optimizer, loss, accuracy, momentum,
                          regularization, batch_size)

    last_dim = num_features
    # Add Layers
    print(topology)
    for num_nodes in topology:
        layer = HiddenLayer(weight_initialization(num_nodes, last_dim),
                            lr.Constant(num_nodes, last_dim, learning_rate),
                            activation())
        model.add_layer(layer)
        last_dim = num_nodes
    output_layer = OutputLayer(weight_initialization(output_dim, last_dim),
                               lr.Constant(output_dim, last_dim,
                                           learning_rate),
                               af.Linear())
    model.add_layer(output_layer)
    print('momentum:', model.momentum_rate)
    return model


if __name__ == '__main__':

    def grid_search(params, dataset, num_features, output_dim, n_threads=5, save_path='grid_results/grid.csv'):
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
            params['batch_size'],
            params['topology'],
            params['loss'],
            params['accuracy'],
            params['optimizer'],
            params['num_epoch']
        ]
        pool = multiprocessing.Pool(multiprocessing.cpu_count()) if n_threads is None else \
            multiprocessing.Pool(n_threads)
        results = multiprocessing.Manager().list()

        start = time.time()
        for model_param in list(itertools.product(*params)):
            model = initialize_model(model_param, num_features, output_dim)
            print("Model:", model)
            pool.apply_async(func=run,
                             args=(model, results, model_param, dataset))

        pool.close()
        pool.join()

        # Sort results according to the accuracy of models

        # l_results = list(results.sort(key=lambda x: x['accuracy_average_vl'], reverse=True))

        # Write to file results obtained
        write_results(results, save_path)

        with open('grid_results/grid_info.txt', 'a') as info_file:
            total_time = time.gmtime(time.time() - start)
            info_file.write("Grid Search ended in {} hour {} minutes {} seconds \n".format(
                total_time.tm_hour, total_time.tm_min, total_time.tm_sec))
        return results[0]

    def write_results(results, file_path):
        """
            Write results obtained by GridSearch in a txt file
            Param:
                file_path(str): path where we will save our results on GridSearch
        """
        with open(file_path, 'w') as result_file:
            writer = csv.writer(result_file)
            writer.writerow(['accuracy_average_vl', 'accuracy_sd_vl', 'average_tr_error_best_vl',
                             'learning_rate', 'regularization', 'momentum', 'activation_hidden',
                             'weight_init', 'topology'])

            for item in results:
                writer.writerow([
                    str(item['accuracy_average_vl']),
                    str(item['accuracy_sd_vl']),
                    str(item['average_tr_error_best_vl']),
                    str(item['model_param'][0]),
                    str(item['model_param'][1]),
                    str(item['model_param'][2]),
                    str(item['model_param'][4]),
                    str(item['model_param'][3]),
                    item['model_param'][7]
                ])
        return None

    #execute grid search
    #grid_search(parameters, dataset, len(train_data[0]), len(train_label[0]))


# BAGGING FINAL RESULTS

def final_model():
    """
        Return the final model for the ML cup obtained with the final grid search
    """

    #model params contains the best hyperparams obtained with the final grid search 
    model_params = [

        [0.01, 0.0005, 0.6, wi.ranged_uniform_initializer, af.TanH, 
            1, (20, 20), 'mean_squared_error', 'euclidean_loss', 'SGD', 500],

        [0.15, 0.001, 0.8, wi.ranged_uniform_initializer, af.Relu, 
            1, (15, 15), 'mean_squared_error', 'euclidean_loss', 'SGD', 500],

        [0.13, 0.0005, 0.6, wi.ranged_uniform_initializer, af.TanH, 
            1, (15,15), 'mean_squared_error', 'euclidean_loss', 'SGD', 500],

        [0.15, 0.001, 0.8, wi.xavier_initializer, af.Relu, 
            1, (15,15), 'mean_squared_error', 'euclidean_loss', 'SGD', 500],

        [0.13, 0.0005, 1.0, wi.ranged_uniform_initializer, af.TanH, 
            1, (15,15), 'mean_squared_error', 'euclidean_loss', 'SGD', 500],

        [0.15, 0.0005, 0.8, wi.ranged_uniform_initializer, af.TanH, 
            1, (10, 5, 5), 'mean_squared_error', 'euclidean_loss', 'SGD', 500],

        [0.13, 0.00075, 1.0, wi.ranged_uniform_initializer, af.Relu, 
            1, (10, 5, 5), 'mean_squared_error', 'euclidean_loss', 'SGD', 500],

        [0.15, 0.001, 1.0, wi.ranged_uniform_initializer, af.Relu, 
            1, (20, 20), 'mean_squared_error', 'euclidean_loss', 'SGD', 500],

        [0.15, 0.0025, 0.8, wi.xavier_initializer, af.Relu, 
            1, (15, 15), 'mean_squared_error', 'euclidean_loss', 'SGD', 500],
    ]

    #Reading and normalizing data from the ML cup 
    #Note that we take 80% for training and 20% for test
    train_data, train_label, test_data, test_label = read_cup_data(
        "dataset/ML-CUP20-TR.csv", 0.8)
    #train_data, train_label, den_data, den_label = normalize_data(
     #   train_data, train_label)
    #test_data, test_label, _, _ = normalize_data(
      #  test_data, test_label, den_data, den_label)

    training_examples = list(zip(train_data, train_label))
    test_examples = list(zip(test_data, test_label))

    model_test = initialize_model(model_params[0], len(train_data[0]), 2)
    report = model_test.fit(training_examples, test_examples)
    report.plot_accuracy()
    #create ensemble object that will contain all the hypothesis
    ensemble = Bagging(len(training_examples))

    # create and add the model to the ensemble

    for model_param in model_params:
        nn = initialize_model(model_param, len(train_data[0]), 2)
        ensemble.add_neural_network(nn)

    # training all the models in the ensemble

    ensemble.fit(training_examples, test_examples)

    # check models performance (denormalizing)

    i = 1
    for model in ensemble.models:
        predicted_training_data = denormalize_data(
            model.predict(train_data), den_label)
        error = metric_functions['euclidean_loss'](
            predicted_training_data,
            denormalize_data(train_label, den_label)
        )
        print("model ", i, ", training: ", error)

        predicted_test_data = denormalize_data(
            model.predict(test_data), den_label)
        error = metric_functions['euclidean_loss'](
            predicted_test_data,
            denormalize_data(test_label, den_label)
        )

        print("model ", i, ", test: ", error)
        i += 1

    # check ensemble performance

    predicted_training_data = denormalize_data(
        ensemble.predict(train_data), den_label)
    error = metric_functions['euclidean_loss'](
        predicted_training_data,
        denormalize_data(train_label, den_label)
    )
    print("ensemble training: ", error)

    predicted_test_data = denormalize_data(
        ensemble.predict(test_data), den_label)
    error = metric_functions['euclidean_loss'](
        predicted_test_data,
        denormalize_data(test_label, den_label)
    )

    print("ensemble test: ", error)

    return ensemble

final_model()
