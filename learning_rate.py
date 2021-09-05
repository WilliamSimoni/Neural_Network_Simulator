"""
    Learning Rate module to represent the Learning Rate possible
    choice in ML models in particular for NN
"""
import numpy as np


class LearningRate():
    """
        Represent a Learning Rate object which represent how to manages
        Learning Rate in our NN model
    """

    def __init__(self, num_unit, num_input, value=0.1):
        self.learning_rates = np.full(
            (num_unit, num_input + 1), value, dtype=np.float)
        self.update_method = self.constant_update_method
        self.current_method_name = 'constant'

    def update(self, epoch):
        """update the learning rate with the current update_method

        Args:
            epoch (int): epoch used to determine how to change the learning_rate
        """
        self.learning_rates = self.update_method(self.learning_rates, epoch)

    def constant_update_method(self, learning_rate, epoch):
        return learning_rate

    def value(self):
        """return cyrrent learning rate

        Returns:
            array: learning rates
        """
        return self.learning_rates

    def current_method(self):
        """return the current method tu update the learning rate

        Returns:
            string: the current method used to update
        """
        return self.current_method_name

    def deepcopy(self):
        if isinstance(self, Constant):
            return Constant(self.learning_rates.shape[0], self.learning_rates.shape[1] - 1, self.learning_rates[0][0])
        elif isinstance(self, timeBasedDecay):
            return timeBasedDecay(self.learning_rates.shape[0], self.learning_rates.shape[1] - 1, self.learning_rates[0][0])
        elif isinstance(self, LinearDecay):
            return LinearDecay(self.learning_rates.shape[0], self.learning_rates.shape[1] - 1, self.learning_rates[0][0])
        else:
            return LearningRate(self.learning_rates.shape[0], self.learning_rates.shape[1] - 1, self.learning_rates[0][0])


class Constant(LearningRate):
    """Learning rate is not updated
    """

    def __init__(self, num_unit, num_input, value=0.1):
        """create a constant learning rate object
        Args:
            num_unit (int): number of unit in the layer
            num_input (int): number of input for each unit
            value (int, optional): the value to which initialize the learning rate. Defaults to 0.1.
        """
        super().__init__(num_unit, num_input, value)


class timeBasedDecay(LearningRate):
    """Learning rate is updated using time based decay
    """

    def __init__(self, num_unit, num_input, value=0.1, decay=0.0001, min_value=0.01):
        """create a time based decay learning rate object
        Args:
            num_unit (int): number of unit in the layer
            num_input (int): number of input for each unit
            value (int, optional): the value to which initialize the learning rate. Defaults to 0.1.
            decay (float): determine how fast the learning rate goes to 0
            min_value (float): min value that the learning rate can reach
        """

        super().__init__(num_unit, num_input, value)

        def _time_based_decay(learning_rates, epoch):
            new_learning_rates = learning_rates * 1./(1. + decay * epoch)
            return new_learning_rates if new_learning_rates[0][0] > min_value else learning_rates

        self.update_method = _time_based_decay
        self.current_method_name = 'time_based_decay'


class LinearDecay(LearningRate):
    """ the learning rate will be updated using a linear decay strategy
    """

    def __init__(self, num_unit, num_input, value=0.1, tau=100, lr_tau=0.01):
        """create a linear decay learning rate object
        Args:
            num_unit (int): number of unit in the layer
            num_input (int): number of input for each unit
            value (int, optional): the value to which initialize the learning rate. Defaults to 0.1.
            tau (int): if num_epoch >= tau then learning_rate = lr_tau. Defaults to 100
            lr_tau (float): learning rate to which arrive when tau = num_epoch. Defaults to 0.01
        """

        super().__init__(num_unit, num_input, value)

        lr0 = value

        def _linear_decay(learning_rates, epoch):
            alpha = (epoch)/tau if tau > epoch else 1
            new_lr = (1 - alpha)*lr0 + alpha*lr_tau
            learning_rates[True] = new_lr
            return learning_rates

        self.update_method = _linear_decay
        self.current_method_name = 'linear_decay'
