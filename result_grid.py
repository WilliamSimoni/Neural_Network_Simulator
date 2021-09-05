import pandas as pd

def result_grid(file_path_1, file_path_2, file_path_3, file_path_4, *args):
    df = pd.read_csv(file_path_1)
    
    if file_path_2 != '' and file_path_3 != '':
        df_2 = pd.read_csv(file_path_2)
        df_3 = pd.read_csv(file_path_3)
        df_4 = pd.read_csv(file_path_4)
        frames = [df, df_2, df_3, df_4]

        df = pd.concat(frames)

    df = df[df['accuracy_average_vl'] < 3]
    learning_rate_grouped = df.groupby(by=df['learning_rate']).agg({'accuracy_average_vl': ['mean', 'std'],
                                                                    'accuracy_sd_vl': ['mean', 'std'],
                                                                    'average_tr_error_best_vl': ['mean', 'std']}).reset_index()
    learning_rate_grouped.columns = ['learning_rate', 'average_accuracy_vl', 'sd_accuracy_vl', 'average_accuracy_sd_vl',
                                     'sd_accuracy_sd_vl', 'average_tr_error_best_vl', 'sd_average_tr_error_best_vl']
    print(df.head())
    print(df.info())
    print("Learning Rate Analysis: ")
    print(learning_rate_grouped.head())

    print("Regularization Analysis: ")
    regularization_grouped = df.groupby(by=df['regularization']).agg({'accuracy_average_vl': ['mean', 'std'],
                                                                    'accuracy_sd_vl': ['mean', 'std'],
                                                                    'average_tr_error_best_vl': ['mean', 'std']}).reset_index()
    regularization_grouped.columns = ['regularization', 'average_accuracy_vl', 'sd_accuracy_vl', 'average_accuracy_sd_vl',
                                     'sd_accuracy_sd_vl', 'average_tr_error_best_vl', 'sd_average_tr_error_best_vl']
    print(regularization_grouped.head())

    print("Momentum Analysis: ")
    momentum_grouped = df.groupby(by=df['momentum']).agg({'accuracy_average_vl': ['mean', 'std'],
                                                                    'accuracy_sd_vl': ['mean', 'std'],
                                                                    'average_tr_error_best_vl': ['mean', 'std']}).reset_index()
    momentum_grouped.columns = ['momentum', 'average_accuracy_vl', 'sd_accuracy_vl', 'average_accuracy_sd_vl',
                                     'sd_accuracy_sd_vl', 'average_tr_error_best_vl', 'sd_average_tr_error_best_vl']

    print(momentum_grouped.head())

    print("Topology Analysis: ")
    topology_grouped = df.groupby(by=df['topology']).agg({'accuracy_average_vl': ['mean', 'std'],
                                                                    'accuracy_sd_vl': ['mean', 'std'],
                                                                    'average_tr_error_best_vl': ['mean', 'std']}).reset_index()
    topology_grouped.columns = ['topology', 'average_accuracy_vl', 'sd_accuracy_vl', 'average_accuracy_sd_vl',
                                     'sd_accuracy_sd_vl', 'average_tr_error_best_vl', 'sd_average_tr_error_best_vl']

    print(topology_grouped.head())

    print("Weight initialization: ")
    weight_grouped = df.groupby(by=df['weight_init']).agg({'accuracy_average_vl': ['mean', 'min'],
                                                                    'accuracy_sd_vl': ['mean', 'min'],
                                                                    'average_tr_error_best_vl': ['mean', 'min']}).reset_index()
    weight_grouped.columns = ['weight_init', 'average_accuracy_vl', 'min_average_sd', 'average_accuracy_sd_vl',
                              'min_sd_accuracy_vl', 'average_tr_error_best_vl', 'min_average_tr_error']

    print(weight_grouped.head())

    print("Activation function Analysis: ")
    activation_grouped = df.groupby(by=df['activation_hidden']).agg({'accuracy_average_vl': ['mean', 'std'],
                                                                    'accuracy_sd_vl': ['mean', 'std'],
                                                                    'average_tr_error_best_vl': ['mean', 'std']}).reset_index()
    activation_grouped.columns = ['activation_hidden', 'average_accuracy_vl', 'sd_accuracy_vl', 'average_accuracy_sd_vl',
                                     'sd_accuracy_sd_vl', 'average_tr_error_best_vl', 'sd_average_tr_error_best_vl']

    print(activation_grouped.head())


result_grid('grid_results/grid_search_8/grid-William-1.csv', '',
            '', '')