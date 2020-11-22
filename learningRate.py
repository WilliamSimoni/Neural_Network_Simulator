import numpy as np
import report as rp

class LearningRate():

    def __init__(self, num_unit, num_input, value=0):
        self.learning_rates = np.full((num_unit, num_input + 1), value, dtype=np.float)
        self.update_method = lambda learning_rates, epoch: learning_rates
        self.current_method_name = 'constant'

    def update(self, epoch):
        """update the learning rate with the current update_method

        Args:
            epoch (int): epoch used to determine how to change the learning_rate
        """
        self.learning_rates = self.update_method(self.learning_rates, epoch)

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

    def constant(self):
        """if called, the learning rate will not be updated
        """
        self.update_method = lambda learning_rates, epoch: learning_rates
        self.current_method_name = 'constant'

    def time_based_decay(self, decay=0.0001):
        """if called, the learning rate will be updated using a time based decay strategy

        Args:
            decay (float): determine how fast the learning rate goes to 0
        """

        def _time_based_decay(learning_rates, epoch):
            return learning_rates * 1./(1. + decay * epoch)

        self.update_method = _time_based_decay
        self.current_method_name = 'time_based_decay'

    def linear_decay(self, tau, lr0, lr_tau):
        """if called, the learning rate will be updated using a linear decay strategy

        Args:
            tau (int): if num_epoch >= tau then learning_rate = lr_tau
            lr0 (float): learning rate from which to start
            lr_tau (float): learning rate to which arrive when tau = num_epoch
        """

        def _linear_decay(learning_rates, epoch):
            alpha = (epoch)/tau if tau > epoch else 1
            print(alpha)
            new_lr = (1 - alpha)*lr0 + alpha*lr_tau
            print(new_lr)
            learning_rates[True] = new_lr
            return learning_rates
        
        self.update_method = _linear_decay
        self.current_method_name = 'linear_decay'


lr = LearningRate(3, 3, 0.1)
#lr.linear_decay(110, 0.1, 0.001)
#comment this and decomment above to try linear decay 
lr.time_based_decay()

report = rp.Report(200)

for i in range(0, 200):
    lr.update(i)
    report.add_training_error(lr.value()[0][0],i)

report.plotLoss()