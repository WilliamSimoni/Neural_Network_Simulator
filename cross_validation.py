import numpy as np

def cross_validation(training_set, num_subsets):

    splitted_training_set = np.array_split(training_set, num_subsets)

    for k in range(0, num_subsets):
        

training_examples = np.array([([1,1],[0]), ([0,1], [1]), ([1,0], [1]), ([0,0], [0])], dtype=object)

cross_validation(training_examples, 3)