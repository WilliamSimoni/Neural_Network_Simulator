"""
    Module utility to define some utilities functions, like read data,
    divide data in training and test set and so on.
"""
import numpy as np
import csv

def read_monk_data(file_path, train_dim=1.0, shuffle=False):
    """
        Read data from Monk dataset to put as input in ML algorithms

        Param:
            file_path (string): Path of Monk dataset
            train_dim (float): number between 0.0 to 1.0 to use as partition 
                               between training and test set.
        Semantics of Monk Dataset:
            1° Row: label of data with value 0, 1
            2° Row: has value 1, 2, 3
            3° Row: has value 1, 2, 3
            4° Row: has value 1, 2
            5° Row: has value 1, 2, 3
            6° Row: has value 1, 2, 3, 4
            7° Row: has value 1, 2
    """
    with open(file_path, "r") as monk_file:
        string_data = [line.split() for line in monk_file]
        label = [int(row[0]) for row in string_data]
        data = []
        
        for row in string_data:
            
            data_row = np.zeros(17)
            data_row[int(row[1]) - 1] = 1
            data_row[int(row[2]) + 2] = 1
            data_row[int(row[3]) + 5] = 1
            data_row[int(row[4]) + 7] = 1
            data_row[int(row[5]) + 10] = 1
            data_row[int(row[6]) + 14] = 1
            data.append(data_row)

    if 0 < train_dim < 1:
        dim_data = int(train_dim * len(data))
        return data[:dim_data], label[:dim_data], data[dim_data:], label[dim_data:] 
      
    return data, label, [], []

def read_cup_data(file_path, train_dim=1.0):
    """
        Read data from ML Cup dataset to put as input in ML algorithms

        Param:
            file_path (string): Path of ML Cup dataset
            train_dim (float): number between 0.0 to 1.0 to use as partition 
                               between training and test set.
    """
    with open(file_path, "r") as cup_file:
        reader = csv.reader(cup_file, delimiter=",")
        labels = []
        data = []
        for row in reader:
            if row[0][0] != '#':
                # Id, 10 data, 2 label
                labels.append([float(row[11]), float(row[12])])
                data.append([float(x) for x in row[1:11]])

    if 0 < train_dim < 1:
        dim_data = int(train_dim * len(data))
        return np.array(data[:dim_data]), np.array(labels[:dim_data]), np.array(data[dim_data:]), np.array(labels[dim_data:]) 
      
    return np.array(data), np.array(labels), [], []

def read_blind_data(file_path):
    """
        Read blind data from ML Cup dataset to set ML algorithms

        Param:
            file_path (string): Path of ML Cup blind dataset
    """
    with open(file_path, "r") as cup_file:
        reader = csv.reader(cup_file, delimiter=",")
        id_name = []
        data = []
        for row in reader:
            if row[0][0] != '#':
                # Id, 10 data, 2 label
                id_name.append(int(row[0]))
                data.append([float(x) for x in row[1:11]])

    return id_name, np.array(data) 