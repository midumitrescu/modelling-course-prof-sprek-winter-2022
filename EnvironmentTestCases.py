import unittest
from model import Environment as Environment
import numpy as np
from numpy.testing import *

class EnvironmentTestCases(unittest.TestCase):
    default_environment = Environment()

    def test_check_for_overflow_on_low_y_1(self):
        overflowing_on_low_y = np.array([[2, 2.1], [-2.9, -3.1]])
        assert_array_equal([[2, 2.1], [-2.9, -2.9]],
                           self.default_environment.correct_for_overflow(overflowing_on_low_y))

    def test_check_for_overflow_on_low_y_2(self):
        overflowing_on_low_y = np.array([[1.9, 2, 2.1], [-2.8, -2.9, -3.1]])
        assert_array_equal([[1.9, 2, 2.1], [-2.8, -2.9, -2.9]],
                           self.default_environment.correct_for_overflow(overflowing_on_low_y))

    def test_check_for_overflow_on_high_y_1(self):
        overflowing_on_high_y = np.array([[2, 2.1], [2.9, 3.1]])
        assert_array_equal([[2, 2.1], [2.9, 2.9]], self.default_environment.correct_for_overflow(overflowing_on_high_y))

    def test_check_for_overflow_on_high_y_2(self):
        overflowing_on_high_y = np.array([[1.9, 2, 2.1], [2.8, 2.9, 3.1]])
        assert_array_equal([[1.9, 2, 2.1], [2.8, 2.9, 2.9]],
                           self.default_environment.correct_for_overflow(overflowing_on_high_y))

    def test_check_for_overflow_on_low_x_1(self):
        overflowing_on_low_x = np.array([[-2.9, -3.1], [2, 2.1]])
        assert_array_equal([[-2.9, -2.9], [2, 2.1]],
                           self.default_environment.correct_for_overflow(overflowing_on_low_x))

    def test_check_for_overflow_on_low_x_2(self):
        overflowing_on_low_x = np.array([[-2.8, -2.9, -3.1], [1.9, 2, 2.1]])
        assert_array_equal([[-2.8, -2.9, -2.9], [1.9, 2, 2.1]],
                           self.default_environment.correct_for_overflow(overflowing_on_low_x))

    def test_check_for_overflow_on_high_x_1(self):
        overflowing_on_high_x = np.array([[2.9, 3.1], [2, 2.1]])
        assert_array_equal([[2.9, 2.9], [2, 2.1]], self.default_environment.correct_for_overflow(overflowing_on_high_x))

    def test_check_for_overflow_on_high_x_2(self):
        overflowing_on_high_x = np.array([[2.9, 2.9, 3.1], [1.9, 2, 2.1]])
        assert_array_equal([[2.9, 2.9, 2.9], [1.9, 2, 2.1]],
                           self.default_environment.correct_for_overflow(overflowing_on_high_x))

    def test_default_x_width(self):
        self.assertEqual(6, self.default_environment.x_width())

    def test_default_y_width(self):
        self.assertEqual(6, self.default_environment.y_width())

if __name__ == '__main__':
    unittest.main()
