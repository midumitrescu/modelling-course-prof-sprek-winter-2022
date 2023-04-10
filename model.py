import random
import unittest
from collections import namedtuple
from numpy.testing import *

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

# define box as the space contained x_ <= x <
from numpy.testing import assert_array_almost_equal

Box_limits = namedtuple('Limits', 'left right')


def width(lims: Box_limits):
    return lims.right - lims.left


class Environment():

    def __init__(self, x_lims=Box_limits(-3, 3), y_lims=Box_limits(-3, 3)):
        self.x_lims = x_lims
        self.y_lims = y_lims

    def x_width(self):
        return width(self.x_lims)

    def y_width(self):
        return width(self.y_lims)

    def correct_for_overflow(self, trajectory, index=None):
        if index is None:
            index = trajectory.shape[1] - 1

        last_x, last_y = trajectory[:, index]
        if not self.x_lims.left < last_x < self.x_lims.right:
            trajectory[0, index] = trajectory[0, index - 1]
        if not self.y_lims.left < last_y < self.y_lims.right:
            trajectory[1, index] = trajectory[1, index - 1]

        return trajectory


default_environent = Environment()


def movement_model(index, noise_mouse):
    return noise_mouse[:, index]


class MouseModel():

    def __init__(self, T=1000, dt=0.1,
                 sigma_movement=np.array([0.1, 0.2]),
                 starting_position=np.zeros(4),
                 environment=default_environent,
                 mating_w=None,
                 sigma_mating=np.array([1, 1])):
        self.T = 1000
        self.dt = 0.1

        self.iterations = int(T / dt)
        self.time = np.linspace(0, T, self.iterations)

        self.x_lims = Box_limits(-3, 3)
        self.y_lims = Box_limits(-3, 3)

        self.sigma_movement = sigma_movement
        self.mouse_1_trajectory = np.zeros((2, self.iterations - 1))
        self.mouse_2_trajectory = np.zeros((2, self.iterations - 1))
        self.mouse_1_trajectory[:, 0] = starting_position[:2]
        self.mouse_2_trajectory[:, 0] = starting_position[2:]
        self.environment = environment

        if mating_w is None:
            mating_w = 2 * np.pi / T

        self.mating_period = np.sin(mating_w * self.time + np.pi / 2)
        self.sigma_mating = sigma_mating

    def simulate_independent_movement(self):

        extended_sigma_movement = np.repeat(self.sigma_movement, 2) ** 2
        covar_scaled_with_sqrt_dt = np.diag(self.dt * extended_sigma_movement)
        self.noise_driven_movement = np.random.multivariate_normal(np.zeros(4),
                                                                   covar_scaled_with_sqrt_dt,
                                                                   self.iterations).T

    def simulate(self):
        self.simulate_independent_movement()

        for index in range(0, len(self.time) - 2):
            mating = self.mating_model(index)
            self.mouse_1_trajectory[:, index + 1] = self.mouse_1_trajectory[:, index] \
                                                    + movement_model(index, self.noise_driven_movement[
                                                                            :2, :])
            self.mouse_2_trajectory[:, index + 1] = self.mouse_2_trajectory[:, index] \
                                                    + movement_model(index, self.noise_driven_movement[2:, :])
            self.environment.correct_for_overflow(self.mouse_1_trajectory, index + 1)
            self.environment.correct_for_overflow(self.mouse_2_trajectory, index + 1)
        return np.vstack((self.mouse_1_trajectory, self.mouse_2_trajectory))

    def mating_model(self, index):

        m_1 = self.mouse_1_trajectory[:, index]
        m_2 = self.mouse_2_trajectory[:, index]
        distance_squared = np.sum((m_1 - m_2) ** 2)

        mating_ = 2 * self.mating_period[index] * (m_1 - m_2) * np.exp(-distance_squared / (self.sigma_mating ** 2))
        mating_strength = mating_
        return np.vstack(
            (mating_strength,
             -mating_strength)
        )


def plot_trajectory(trajectory, fig=plt, ax=plt.gca(), color='blue', label='Mouse', show=True,
                    environment=default_environent):
    ax.plot(trajectory[0], trajectory[1], color=color, label=label)
    ax.set_aspect("equal")
    ax.legend()

    rect = patches.Rectangle((environment.x_lims.left - .05, environment.y_lims.left - .05),
                             environment.x_width() + 0.1,
                             environment.y_width() + 0.1, linewidth=3, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    if show:
        fig.show()


class ModelTestCases(unittest.TestCase):
    default_model = MouseModel()

    def test_size_of_time_array_is_as_expected(self):
        self.assertEqual(10000, self.default_model.time.size)

    def test_forward_euler_working(self):
        pass

    def test_print_box(self):
        x = np.linspace(-3, 3, self.default_model.iterations)
        trajectory = np.vstack((x, x))

        fig, ax = plt.subplots(1, 1)

        plot_trajectory(trajectory, fig, ax)

    def test_check_movement_inside_bounds_(self):
        x = np.linspace(-2, 1, 100)
        y = -4 - x
        trajectory = np.vstack((x, y))

        fig, ax = plt.subplots(1, 1)
        plot_trajectory(trajectory, fig, ax)

        overflow = trajectory[:, :35]

        last_x, last_y = overflow[:, -1]

    def test_inputting_a_start_vector_makes_mouse_start_there(self):
        model = MouseModel(starting_position=(np.array([1, 1, 0, 0])))
        trajectory = model.simulate()
        assert_array_equal([1, 1], trajectory[:2, 0])

    def test_forward_euler_with_noise(self):
        np.random.seed(0)

        mouse_model = MouseModel()

        trajectory = mouse_model.simulate()
        plot_trajectory(trajectory[:2], color='red', label='Mouse 1', show=False)
        plot_trajectory(trajectory[2:], color='green', label='Mouse 2', show=True)

    def test_forward_euler_with_noise_bigger_env(self):
        np.random.seed(0)

        mouse_model = MouseModel()

        trajectory = mouse_model.simulate()
        bigger_env = Environment(x_lims=Box_limits(-4, 4), y_lims=Box_limits(-4, 4))
        plot_trajectory(trajectory[:2], color='red', label='Mouse 1', show=False, environment=bigger_env)
        plot_trajectory(trajectory[2:], color='green', label='Mouse 2', show=True, environment=bigger_env)

    def test_mating_period_function_1(self):
        plt.plot(self.default_model.time, self.default_model.mating_period)
        plt.show()

    def test_mating_period_function_2(self):
        test_model = MouseModel(mating_w=4 * np.pi / self.default_model.T)
        plt.plot(test_model.time, test_model.mating_period)
        plt.show()

    def test_low_sigma_movement_implies_no_movement(self):
        test_model = MouseModel(sigma_movement=np.array([10 ** -6, 10 ** -6]), starting_position=(np.array([-2, -2, 2, 2])))
        trajectory = test_model.simulate()

        np.apply_along_axis(lambda array: assert_array_almost_equal([-2, -2, 2, 2], array, decimal=4,
                                                                    err_msg='Mice should not move with such low sigma'),
                            arr=trajectory, axis=0)

        plot_trajectory(trajectory[:2], show=False, label='Mouse 1')
        plot_trajectory(trajectory[2:], color='green', label='Mouse 2', show=True)

    def test_movement_model(self):
        np.random.seed(0)
        test_model = MouseModel(starting_position=(np.array([-2, -2, 2, 2])),
                                mating_w=0)
        trajectory = test_model.simulate()

        assert_array_almost_equal(np.array([
            [-2.00000, -1.96905, -1.93901, -1.93445, -1.92041, -1.91051, -1.88318, -1.88173, -1.87683, -1.88783],
            [-2.00000, -1.92914, -1.93392, -1.88793, -1.87738, -1.90439, -1.92786, -1.93378, -1.92182, -1.91688],
            [2.00000, 2.11157, 2.22968, 2.22316, 2.27129, 2.36578, 2.20432, 2.34787, 2.44481, 2.38866],
            [2.00000, 2.02531, 1.96350, 1.98947, 1.99716, 1.98419, 2.02553, 1.93354, 2.02647, 1.90120]]),
            trajectory[:, :10], decimal=5)

    def test_movement_variance(self):
        np.random.seed(0)
        test_model = MouseModel()
        test_model.simulate_independent_movement()
        movement = test_model.noise_driven_movement
        assert_array_almost_equal([0.001, 0.001, 0.004, 0.004],
                                  np.var(movement, axis=1), decimal=3)

    def test_mating_function(self):
        no_noise_model = MouseModel(sigma_movement=np.array([10 ** -8, 10 ** -8]))
        no_noise_model.mating_model(index=0)

