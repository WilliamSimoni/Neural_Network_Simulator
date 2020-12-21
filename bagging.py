import math
import numpy as np
from random import choice
from neural_network import NeuralNetwork
from utility import normalize_data, read_cup_data


class Bagging():

    def __init__(self , sample_size):
        self.sample_size = sample_size
        self.models = []
    
    def _generate_sample(self, dataset):
        return [choice(dataset) for i in range(0, self.sample_size)]
    
    def add_model(self, model):
        self.models.append(model)
    
    def training(self, dataset):
        for model in self.models:
            model.fit(self._generate_sample(dataset))
    
    def predict(self, input):
        return np.mean([model.predict(input) for model in self.models], axis=0)