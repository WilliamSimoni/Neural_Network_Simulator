"""
    Module test_loss test the loss Module
"""
import unittest
import numpy as np
from loss import *

class TestLoss(unittest.TestCase):
    """
        TestLoss is an unittest Class to test the Loss module
    """

    def test_euclidean_loss(self):
        """
            Test the euclidean loss with valid values
        """
        self.assertEqual(1, euclidean_loss(np.array([2]), np.array([3])))
        self.assertEqual(3, euclidean_loss(np.array([[1, 3, 3]]), np.array([[0, 1, 1]])))

    def test_invalid_euclidean_loss(self):
        """
            Test the euclidean loss with invalid values in input
        """
        try:
            self.assertRaises(ValueError, euclidean_loss(np.array([1]), [1]))
            self.assertRaises(ValueError, euclidean_loss([1], np.array([1])))
            self.assertRaises(ValueError, euclidean_loss(np.array([1]), np.array([1])))
        except:
            pass
        
    