"""
    Report module used to report result of our NN simulator
"""
import matplotlib.pyplot as plt
import numpy as np

class Report():
    """
        Report class is used to report result of NN simulator
    """
    def __init__(self, max_epochs):
        if max_epochs > 0:
            self.training_error = np.zeros(max_epochs)
        else:
            raise ValueError

    def add_training_error(self, error, num_epochs):
        if num_epochs < 0 or num_epochs > self.training_error.size:
            raise ValueError
        self.training_error[num_epochs] = error

    def plotLoss(self):
        plt.plot(self.training_error)
        plt.ylabel('training error')
        plt.show()
        plt.savefig('training_error.png')