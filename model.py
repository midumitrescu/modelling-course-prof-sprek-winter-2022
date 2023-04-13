import random
import unittest
from abc import abstractmethod
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


class Experiment:

    def __init__(self, T=1000, dt=0.1):
        self.T = 1000  # seconds
        self.dt = 0.1

        self.iterations = int(T / dt)
        self.time = np.linspace(0, T, self.iterations)

        self.x_lims = Box_limits(-3, 3)
        self.y_lims = Box_limits(-3, 3)
        self.current_index = 0

    def step(self):
        self.current_index += 1

    def next_index(self):
        return self.current_index + 1

    def reset(self):
        self.current_index = 0

    @property
    def current_time(self):
        return self.time[self.current_index]


default_experiment = Experiment()


class Model:
    def __init__(self, experiment: Experiment = default_experiment, environment: Environment = default_environent,
                 starting_position=np.zeros(4), active=True):
        self._experiment = experiment
        self._environment = environment
        self.mice_trajectory = np.zeros((4, self.experiment.iterations))
        self.all_gradients = np.zeros((4, self.experiment.iterations))
        self.mice_trajectory[:, 0] = starting_position

    @property
    def experiment(self):
        return self._experiment

    @property
    def environment(self):
        return self._environment

    def simulate(self):
        self.experiment.reset()

        for index in range(0, len(self.experiment.time) - 1):
            gradient = self.gradient(self.mice_trajectory[:, self.experiment.current_index])
            self.mice_trajectory[:, self.experiment.current_index + 1] = \
                self.mice_trajectory[:, self.experiment.current_index] + \
                gradient

            self.environment.correct_for_overflow(self.mice_trajectory[:2], self.experiment.next_index())
            self.environment.correct_for_overflow(self.mice_trajectory[2:], self.experiment.next_index())

            self.save_gradient(gradient)
            self.experiment.step()
        return self.mice_trajectory

    @abstractmethod
    def gradient(self, mice_position):
        pass

    def save_gradient(self, gradient):
        self.all_gradients[:, self.experiment.current_index] = gradient


class Mating_Model(Model):

    def __init__(self, experiment: Experiment = default_experiment, mating_w=None, sigma_mating=np.array([2, 2]),
                 mating_peak=np.array([0.005, 0.005])):
        super().__init__(experiment)
        if mating_w is None:
            mating_w = 2 * np.pi / experiment.T

        self.mating_period = np.sin(mating_w * experiment.time)
        self.sigma_mating = sigma_mating
        self.mating_peak = mating_peak
        self.all_mating_gradients = np.zeros((4, experiment.iterations - 1))

    def gradient(self, mice_position):
        m_1 = mice_position[:2]
        m_2 = mice_position[2:]
        distance_squared = np.sum((m_1 - m_2) ** 2)

        mating_strength = -1 * self.mating_peak * \
                          2 * self.mating_period[self.experiment.current_index] * \
                          (m_1 - m_2) * \
                          np.exp(-distance_squared / (self.sigma_mating ** 2))

        mating_gradient = np.hstack((mating_strength, -mating_strength)).T
        self.all_mating_gradients[:, self.experiment.current_index] = mating_gradient
        return mating_gradient


default_mating_model = Mating_Model()


class Movement_Model(Model):

    def __init__(self, experiment: Experiment = default_experiment,
                 sigma_movement=np.array([0.05, 0.1]), testing=True):
        super().__init__(experiment)
        self.sigma_movement = sigma_movement

        if testing:
            np.random.seed(0)

        self.simulate_independent_movement()

    def simulate_independent_movement(self):
        extended_sigma_movement = np.repeat(self.sigma_movement, 2) ** 2
        covar_scaled_with_sqrt_dt = np.diag(self.experiment.dt * extended_sigma_movement)
        self.noise_driven_movement = np.random.multivariate_normal(np.zeros(4),
                                                                   covar_scaled_with_sqrt_dt,
                                                                   self.experiment.iterations).T

    def gradient(self, mice_position):
        return self.noise_driven_movement[:, self.experiment.current_index]


default_movement_model = Movement_Model()


def sigmoid(x, half_value=100, max: float = 1, slope=25):
    return max / (1 + np.exp(- (x - half_value) / slope))


class Feeding_Model(Model):

    def __init__(self, environment: Environment = default_environent, experiment: Experiment = default_experiment,
                 food_position: np.ndarray = None, feeding_radius=10 ** -3,
                 hunger_freq: np.ndarray = np.array([100, 100])):
        super().__init__(experiment, environment)

        self.food_pos = self.init_food_position(food_position)
        self.mouse_1_feed_events = list([-100])
        self.mouse_2_feed_events = list([-100])
        self.feeding_radius = feeding_radius
        self.hunger_freq = hunger_freq

    def init_food_position(self, food_position):
        if food_position is not None:
            return food_position

        x_pos = self.environment.x_width() * (np.random.random() - 0.5)
        y_pos = self.environment.y_width() * (np.random.random() - 0.5)
        return np.array([x_pos, y_pos])

    def check_if_mice_are_feeding(self, mice_position):
        if np.linalg.norm(mice_position[:2] - self.food_pos) < self.feeding_radius:
            self.mouse_1_feed_events.append(self.experiment.current_time)
            print(f'Mouse 1 has fed at time {self.experiment.current_time}')
        if np.linalg.norm(mice_position[2:] - self.food_pos) < self.feeding_radius:
            self.mouse_2_feed_events.append(self.experiment.current_time)
            print(f'Mouse 2 has fed at time {self.experiment.current_time}')

    def gradient(self, mice_position):

        self.check_if_mice_are_feeding(mice_position)
        last_feeding = np.array([self.mouse_1_feed_events[-1], self.mouse_2_feed_events[-1]])
        next_ideal_feeding_time = last_feeding + self.hunger_freq

        return -1 * (mice_position - np.tile(self.food_pos, 2)) * np.repeat(sigmoid(next_ideal_feeding_time, max=0.1),
                                                                            2)


default_feeding_model = Feeding_Model()
no_feeding = type("No_Feeding_Model", (Feeding_Model, object), {"gradient": lambda self, _: np.zeros(4)})()


class Mouse_Model(Model):

    def __init__(self, experiment: Experiment = default_experiment, starting_position=np.zeros(4),
                 environment=default_environent, movement_model: Movement_Model = default_movement_model,
                 mating_model: Mating_Model = default_mating_model,
                 feeding_model: Feeding_Model = default_feeding_model):
        super().__init__(experiment, environment, starting_position)

        self.mating_model = mating_model
        self.movement_model = movement_model
        self.feeding_model = feeding_model

    def gradient(self, mice_position):
        return self.movement_model.gradient(mice_position) + \
               self.mating_model.gradient(mice_position) + \
               self.feeding_model.gradient(mice_position)


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


class Model_Test_Cases(unittest.TestCase):
    default_model = Mouse_Model()

    def test_size_of_time_array_is_as_expected(self):
        self.assertEqual(10000, default_experiment.time.size)

    def test_print_box(self):
        x = np.linspace(-3, 3, default_experiment.iterations)
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
        model = Mouse_Model(starting_position=(np.array([1, 1, 0, 0])))
        trajectory = model.simulate()
        assert_array_equal([1, 1], trajectory[:2, 0])

    def test_forward_euler_with_noise(self):
        np.random.seed(0)

        mouse_model = Mouse_Model()

        fig, ax = plt.subplots(1, 3)
        fig.set_size_inches(18.5, 10.5)

        trajectory = mouse_model.simulate()
        plot_trajectory(trajectory[:2], fig, ax[0], color='red', label='Mouse 1', show=False)
        plot_trajectory(trajectory[2:], fig, ax[0], color='green', label='Mouse 2', show=False)

        plot_trajectory(trajectory[:2, :5000], fig, ax[1], color='red', label='Mouse 1, mating on', show=False)
        plot_trajectory(trajectory[2:, :5000], fig, ax[1], color='green', label='Mouse 2, mating on', show=False)

        plot_trajectory(trajectory[:2, 5000:], fig, ax[2], color='red', label='Mouse 1, mating off', show=False)
        plot_trajectory(trajectory[2:, 5000:], fig, ax[2], color='green', label='Mouse 2, mating off')

    def test_forward_euler_with_noise_bigger_env(self):
        np.random.seed(0)

        mouse_model = Mouse_Model()

        trajectory = mouse_model.simulate()
        bigger_env = Environment(x_lims=Box_limits(-4, 4), y_lims=Box_limits(-4, 4))
        plot_trajectory(trajectory[:2], color='red', label='Mouse 1', show=False, environment=bigger_env)
        plot_trajectory(trajectory[2:], color='green', label='Mouse 2', show=True, environment=bigger_env)

    def test_mating_period_function_1(self):
        plt.plot(default_experiment.time, default_mating_model.mating_period)
        plt.show()

    def test_mating_period_function_2(self):
        quicker_mating_model = Mating_Model(mating_w=4 * np.pi / default_experiment.T)
        plt.plot(default_experiment.time, quicker_mating_model.mating_period)
        plt.show()

    def test_low_sigma_movement_implies_no_movement(self):
        no_mating = Mating_Model(mating_peak=np.zeros(2))
        very_low_movement = Movement_Model(sigma_movement=np.array([10 ** -6, 10 ** -6]))
        test_model = Mouse_Model(movement_model=very_low_movement,
                                 starting_position=(np.array([-2, -2, 2, 2])),
                                 mating_model=no_mating,
                                 feeding_model=no_feeding)
        trajectory = test_model.simulate()

        np.apply_along_axis(lambda array: assert_array_almost_equal([-2, -2, 2, 2], array, decimal=4,
                                                                    err_msg='Mice should not move with such low sigma'),
                            arr=trajectory, axis=0)

        plot_trajectory(trajectory[:2], show=False, label='Mouse 1')
        plot_trajectory(trajectory[2:], color='green', label='Mouse 2', show=True)

    def test_movement_model(self):
        np.random.seed(0)
        no_mating = Mating_Model(mating_peak=np.zeros(2))
        somewhat_high_movement_model = Movement_Model(sigma_movement=np.array([0.1, 0.2]))
        test_model = Mouse_Model(starting_position=(np.array([-2, -2, 2, 2])),
                                 movement_model=somewhat_high_movement_model,
                                 environment=default_environent,
                                 mating_model=no_mating,
                                 feeding_model=no_feeding
                                 )
        trajectory = test_model.simulate()

        assert_array_almost_equal(np.array([
            [-2.00000, -1.96905, -1.93901, -1.93445, -1.92041, -1.91051, -1.88318, -1.88173, -1.87683, -1.88783],
            [-2.00000, -1.92914, -1.93392, -1.88793, -1.87738, -1.90439, -1.92786, -1.93378, -1.92182, -1.91688],
            [2.00000, 2.11157, 2.22968, 2.22316, 2.27129, 2.36578, 2.20432, 2.34787, 2.44481, 2.38866],
            [2.00000, 2.02531, 1.96350, 1.98947, 1.99716, 1.98419, 2.02553, 1.93354, 2.02647, 1.90120]]),
            trajectory[:, :10], decimal=5)

    def test_movement_variance(self):
        np.random.seed(0)
        somewhat_high_movement_model = Movement_Model(sigma_movement=np.array([0.1, 0.2]))
        somewhat_high_movement_model.simulate_independent_movement()
        movement = somewhat_high_movement_model.noise_driven_movement
        assert_array_almost_equal([0.001, 0.001, 0.004, 0.004],
                                  np.var(movement, axis=1), decimal=3)

    def test_mating_function_attraction_on(self):
        always_mating = Mating_Model(sigma_mating=np.array([2.5, 2.5]),
                                     mating_peak=np.array([0.005, 0.005]))
        always_mating.mating_period = np.ones(default_experiment.time.shape)
        very_low_movement = Movement_Model(sigma_movement=np.array([10 ** -8, 10 ** -8]))
        no_noise_model = Mouse_Model(movement_model=very_low_movement,
                                     starting_position=np.array([-2, -2, 2, 2]),
                                     mating_model=always_mating
                                     )

        trajectory = no_noise_model.simulate()

        plot_trajectory(trajectory[:2], show=False, label='Mouse 1')
        plot_trajectory(trajectory[2:], color='green', label='Mouse 2', show=True)

        fig, ax = plt.subplots(1, 1)
        ax.plot(default_experiment.time[1:], np.linalg.norm(always_mating.all_mating_gradients[:2, :], axis=0),
                label='Mating desire of Mouse 1')
        ax.plot(default_experiment.time[1:], np.linalg.norm(always_mating.all_mating_gradients[2:, :], axis=0),
                label='Mating desire of Mouse 2')
        fig.legend()
        fig.show()

    def test_mating_function_repulsion_on(self):
        never_mating = Mating_Model(sigma_mating=np.array([2.5, 2.5]),
                                    mating_peak=np.array([0.005, 0.005]))
        never_mating.mating_period = -1 * np.ones(default_experiment.time.shape)

        very_low_movement = Movement_Model(sigma_movement=np.array([10 ** -8, 10 ** -8]))
        no_noise_model = Mouse_Model(movement_model=very_low_movement,
                                     starting_position=np.array([-0.1, -0.1, 0.1, 0.1]),
                                     mating_model=never_mating)

        trajectory = no_noise_model.simulate()

        plot_trajectory(trajectory[:2], show=False, label='Mouse 1')
        plot_trajectory(trajectory[2:], color='green', label='Mouse 2', show=True)

        fig, ax = plt.subplots(1, 1)
        ax.plot(default_experiment.time[1:], np.linalg.norm(never_mating.all_mating_gradients[:2, :], axis=0),
                label='Mating desire of Mouse 1')
        ax.plot(default_experiment.time[1:], np.linalg.norm(never_mating.all_mating_gradients[2:, :], axis=0),
                label='Mating desire of Mouse 2')
        fig.legend()
        fig.show()

    def test_how_seed_works(self):
        np.random.seed(0)
        first = Movement_Model()
        np.random.seed(0)
        second = Movement_Model()
        assert_array_almost_equal(first.noise_driven_movement, second.noise_driven_movement)


class Feeding_Model_Test_Cases(unittest.TestCase):
    def test_feeding_position_initialization(self):
        for initialization in map(lambda _: Feeding_Model().food_pos, range(0, 10000)):
            self.assertTrue(-3 <= initialization[0] <= 3)
            self.assertTrue(-3 <= initialization[1] <= 3)

    def test_feeding_model_in_isolation(self):
        feeding_model = Feeding_Model(food_position=np.array([0, 0]))
        feeding_model.mice_trajectory[:, 0] = [-2, -2, 1, 1]

        trajectory = feeding_model.simulate()
        plot_trajectory(trajectory[:2], show=False, label='Mouse 1')
        plot_trajectory(trajectory[2:], color='green', label='Mouse 2', show=True)
        plt.show()

        plt.plot(default_experiment.time, feeding_model.mice_trajectory[0], color='red', label='Mouse 1 x')
        plt.plot(default_experiment.time, feeding_model.mice_trajectory[1], color='blue', label='Mouse 1 y')
        plt.plot(default_experiment.time, feeding_model.mice_trajectory[2], color='green', label='Mouse 2 x')
        plt.plot(default_experiment.time, feeding_model.mice_trajectory[3], color='yellow', label='Mouse 2 y')
        plt.ylabel("Trajectory")
        plt.legend()
        plt.show()

    def test_sigmoid_implementation_default_values(self):
        x = np.linspace(0, 300, 300)
        z = sigmoid(x)

        plt.plot(x, z)
        plt.xlabel("x")
        plt.ylabel("Sigmoid(X)")
        plt.show()

    def test_sigmoid_implementation_1(self):
        x = np.linspace(0, 200, 400)
        z = sigmoid(x, half_value=50, max=50)
        self.assertAlmostEqual(25, z[100], places=0)

        plt.plot(x, z)
        plt.xlabel("x")
        plt.ylabel("Sigmoid(X)")
        plt.show()

    def test_sigmoid_implementation_2(self):
        x = np.linspace(-100, 300, 400)
        z_default = sigmoid(x, half_value=100, max=50, slope=10)
        self.assertTrue(z_default[199] < 25 < z_default[200])
        z_steep = sigmoid(x, half_value=100, max=50, slope=5)
        z_flat = sigmoid(x, half_value=100, max=50, slope=50)

        plt.plot(x, z_default, color='red', label='Default slope')
        plt.plot(x, z_steep, color='blue', label='Steep slope')
        plt.plot(x, z_flat, color='green', label='Flat slope')
        plt.xlabel("x")
        plt.ylabel("Sigmoid(X)")
        plt.legend()

        plt.show()
