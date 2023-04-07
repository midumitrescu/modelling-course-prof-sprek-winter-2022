import unittest
from collections import namedtuple
from numpy.testing import *

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

T = 1000
dt = 0.1

iterations = int(T / dt)
time = np.linspace(0, T, iterations)

# define box as the space contained x_ <= x <

Lims = namedtuple('Limits', 'left right')
x_lims = Lims(-3, 3)
y_lims = Lims(-3, 3)


def plot_trajectory(fig, ax, trajectory):
    ax.plot(trajectory[0], trajectory[1])
    ax.set_aspect("equal")

    rect = patches.Rectangle((-3.05, -3.05), 6.1, 6.1, linewidth=3, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    fig.show()


def correct_for_overflow(trajectory, index=None):
    if index is None:
        index = trajectory.shape[0] - 1

    last_x, last_y = trajectory[index]
    if not x_lims.left < last_x < x_lims.right:
        trajectory[index, 0] = trajectory[index - 1, 0]
    if not y_lims.left < last_y < y_lims.right:
        trajectory[index, 1] = trajectory[index - 1, 1]

    return trajectory


class ModelTestCases(unittest.TestCase):
    def test_size_of_time_array_is_as_expected(self):
        self.assertEqual(10000, time.size)

    def test_forward_euler_working(self):
        pass

    def test_print_box(self):
        x = np.linspace(-3, 3, iterations)
        trajectory = np.vstack((x, x))

        fig, ax = plt.subplots(1, 1)

        plot_trajectory(fig, ax, trajectory)

    def test_check_movement_inside_bounds_(self):
        x = np.linspace(-2, 1, 100)
        y = -4 - x
        trajectory = np.vstack((x, y))

        fig, ax = plt.subplots(1, 1)
        plot_trajectory(fig, ax, trajectory)

        overflow = trajectory[:, :35]

        last_x, last_y = overflow[:, -1]

    def test_check_for_overflow_on_low_y_1(self):
        overflowing_on_low_y = np.array([[2, -2.9], [2.1, -3.1]])
        assert_array_equal([[2, -2.9], [2.1, -2.9]], correct_for_overflow(overflowing_on_low_y))

    def test_check_for_overflow_on_low_y_2(self):
        overflowing_on_low_y = np.array([[1.9, -2.8], [2, -2.9], [2.1, -3.1]])
        assert_array_equal([[1.9, -2.8], [2, -2.9], [2.1, -2.9]], correct_for_overflow(overflowing_on_low_y))

    def test_check_for_overflow_on_high_y_1(self):
        overflowing_on_high_y = np.array([[2, 2.9], [2.1, 3.1]])
        assert_array_equal([[2, 2.9], [2.1, 2.9]], correct_for_overflow(overflowing_on_high_y))

    def test_check_for_overflow_on_high_y_2(self):
        overflowing_on_high_y = np.array([[1.9, 2.8], [2, 2.9], [2.1, 3.1]])
        assert_array_equal([[1.9, 2.8], [2, 2.9], [2.1, 2.9]], correct_for_overflow(overflowing_on_high_y))


    def test_check_for_overflow_on_low_x_1(self):
        overflowing_on_low_x = np.array([[-2.9, 2], [-3.1, 2.1]])
        assert_array_equal([[-2.9, 2], [-2.9, 2.1]], correct_for_overflow(overflowing_on_low_x))

    def test_check_for_overflow_on_low_x_2(self):
        overflowing_on_low_x = np.array([[-2.8, 1.9], [ -2.9, 2], [-3.1, 2.1]])
        assert_array_equal([[-2.8, 1.9], [ -2.9, 2], [-2.9, 2.1 ]], correct_for_overflow(overflowing_on_low_x))

    def test_check_for_overflow_on_high_x_1(self):
        overflowing_on_high_x = np.array([[2.9, 2], [3.1, 2.1]])
        assert_array_equal([[2.9, 2], [2.9, 2.1]], correct_for_overflow(overflowing_on_high_x))

    def test_check_for_overflow_on_high_x_2(self):
        overflowing_on_high_x = np.array([[1.9, 2.8], [2, 2.9], [2.1, 3.1]])
        assert_array_equal([[1.9, 2.8], [2, 2.9], [2.1, 2.9]], correct_for_overflow(overflowing_on_high_x))

if __name__ == '__main__':
    unittest.main()
