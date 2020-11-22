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
            self.validation_error = np.zeros(max_epochs)
            self.test_error = np.zeros(max_epochs)
        else:
            raise ValueError

    def add_training_error(self, error, num_epoch):
        """
            Add Training Error to use for report and plot
            Param:
                error(float): training error
                num_epoch(int): num of epoch
        """
        if num_epoch < 0 or num_epoch >= self.training_error.size:
            raise ValueError
        self.training_error[num_epoch] = error

    def add_validation_error(self, error, num_epoch):
        """
            Add Training Error to use for report and plot
            Param:
                error(float): training error
                num_epoch(int): num of epoch
        """
        if num_epoch < 0 or num_epoch >= self.validation_error.size:
            raise ValueError
        self.validation_error[num_epoch] = error

    def add_test_error(self, error, num_epoch):
        """
            Add Training Error to use for report and plot
            Param:
                error(float): test error
                num_epoch(int): num of epoch
        """
        if num_epoch < 0 or num_epoch >= self.test_error.size:
            raise ValueError
        self.test_error[num_epoch] = error

    def plot_loss(self):
        """
            Plot Loss error for Training, Validation and Test set
        """
        plt.plot(self.training_error)
        plt.plot(self.validation_error)
        plt.ylabel('Error')
        plt.show()
        plt.savefig('training_error.png')
