import matplotlib.pyplot as plt
import numpy as np

class Report():

    def __init__(self, max_epochs):
        self.training_error = np.zeros(max_epochs)

    def add_training_error(self, error, num_epochs):
        self.training_error[num_epochs] = error

    def plotLoss(self):
        plt.plot(self.training_error)
        plt.ylabel('training error')
        plt.show()
