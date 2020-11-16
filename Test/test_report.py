"""
    Test Report test the report module
"""
import unittest
import numpy as np
from report import Report

class TestReport(unittest.TestCase):
    """
        Test Report class from report module
    """

    def setUp(self):
        """
            Create a report file with 100 as max_epochs
        """
        self.report = Report(100)

    def test_report_constructor(self):
        """
            Test report constructor 
        """
        self.assertEqual(100, len(self.report.training_error))
    
    def test_invalid_report_constructor(self):
        """
            Test invalid Report initialization
        """
        try:
            self.assertRaises(ValueError, Report(-1))
        except:
            pass

    def test_add_training_error(self):
        """
            Test add_training_error of Report class
        """
        self.report.add_training_error(np.array([0.1, 0.1]), 0)
        self.assertEqual([0.1, 0.1], list(self.report.training_error[0]))
        
    def test_invalid_add_training_error(self):
        """
            Test add_training_error with invalid type parameters
        """
        try:
            self.assertRaises(ValueError, self.report.add_training_error(np.array([0.1, 0.1]), -1))
        except:
            pass
        try:
            self.assertRaises(ValueError, self.report.add_training_error(np.array([0.1, 0.1]), 100))
        except:
            pass