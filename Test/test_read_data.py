"""
    Test utility function for read data from monk and ML_CUP datasets
"""
import unittest
import numpy as np
from utility import *

class TestReadData(unittest.TestCase):
    """
        Test Read data functions defined in utility module
    """

    def setUp(self):
        """
            Define path for monk dataset, ML_CUP dataset and blink_ML_CUP dataset
        """
        self.monkPath = "dataset/monks-1.train"
        self.blindPath = "dataset/ML-CUP20-TS.csv"
        self.mlTrainPath = "dataset/ML-CUP20-TR.csv"
    
    def test_monk_data_train(self):
        """
            Test read_monk_data function with all dataset used as training
        """
        train_data, train_label, _, _ = read_monk_data(self.monkPath)
        self.assertEqual(124, len(train_data))
        self.assertEqual(len(train_data), len(train_label))
        self.assertEqual(1, train_label[0])
        self.assertEqual(0, train_label[10])
        self.assertListEqual([1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                             list(train_data[0]))
        self.assertListEqual([1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                             list(train_data[10]))            

    def test_monk_data_validation(self):
        """
            Test read_monk_data function used as training and validation 
        """
        train_data, train_label, valid_data, valid_label = read_monk_data(self.monkPath, 0.8)
        len_data = len(train_data) + len(valid_data)
        self.assertEqual(124, len_data)
        self.assertEqual(len(train_data), len(train_label))
        self.assertEqual(len(valid_data), len(valid_label))
        self.assertEqual(np.round(0.2 * 124), len(valid_data))

    def test_ml_cup_train(self):
        """
            Test read_cup_data function with all dataset used as training
        """
        train_data, train_label, _, _ = read_cup_data(self.mlTrainPath)
        self.assertEqual(1524, len(train_data))
        self.assertEqual(len(train_data), len(train_label))
        self.assertEqual([58.616635, -36.878797], list(train_label[0]))
        
        self.assertListEqual([-1.227729,0.740105,0.453528,-0.761051,-0.537705,1.471803,-1.143195,2.034647,1.603978,-1.399807],
                             list(train_data[0]))
        
    def test_ml_cup_validation(self):
        """
            Test read_cup_data function used as training and validation
        """
        train_data, train_label, valid_data, valid_label = read_cup_data(self.mlTrainPath, 0.8)
        self.assertEqual(1524, len(train_data) + len(valid_data))
        self.assertEqual(len(train_data), len(train_label))
        self.assertEqual(len(valid_data), len(valid_label))
        self.assertEqual(np.round(0.2 * 1524), len(valid_data))

    def test_ml_cup_blind(self):
        """
            Test read_blind_data function with all dataset used as testing set
        """
        blind_id, blind_data = read_blind_data(self.blindPath)
        self.assertEqual(472, len(blind_data))
        self.assertEqual(len(blind_id), len(blind_data))
        