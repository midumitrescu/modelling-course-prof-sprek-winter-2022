import random
import unittest
from abc import abstractmethod
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

Box_limits = namedtuple('Limits', 'left right')


def width(lims: Box_limits):
    return lims.right - lims.left


class Environment():

    def __init__(self, x_lims=Box_limits(-3, 3), y_lims=Box_limits(-3, 3), resolution=100):
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.resolution = resolution

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

    @property
    def x(self):
        return self.limit_as_linspace(self.x_lims)

    @property
    def y(self):
        return self.limit_as_linspace(self.y_lims)

    def limit_as_linspace(self, lims: Box_limits):
        return np.linspace(lims.left, lims.right, self.resolution)

    def random_vector_inside_bounds(self):
        x_pos = self.x_width() * (np.random.random() - 0.5)
        y_pos = self.y_width() * (np.random.random() - 0.5)
        return np.array([x_pos, y_pos])


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
                 starting_position=np.zeros(4)):
        self._experiment = experiment
        self._environment = environment
        self.mice_trajectory = np.zeros((4, self.experiment.iterations))
        self.all_gradients = np.zeros((4, self.experiment.iterations))

        if starting_position is None:
            starting_position = np.hstack((self.environment.random_vector_inside_bounds(),
                                           self.environment.random_vector_inside_bounds()))

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
            gradient = self.gradient_step(self.mice_trajectory[:, self.experiment.current_index])
            self.mice_trajectory[:, self.experiment.current_index + 1] = \
                self.mice_trajectory[:, self.experiment.current_index] + \
                gradient

            self.environment.correct_for_overflow(self.mice_trajectory[:2], self.experiment.next_index())
            self.environment.correct_for_overflow(self.mice_trajectory[2:], self.experiment.next_index())

            self.save_gradient(gradient)
            self.experiment.step()
        return self.mice_trajectory

    @abstractmethod
    def gradient_step(self, mice_position):
        pass

    def save_gradient(self, gradient):
        self.all_gradients[:, self.experiment.current_index] = gradient


class Mating_Model(Model):

    def __init__(self, experiment: Experiment = default_experiment, mating_w=None, sigma_mating=np.array([2, 2]),
                 mating_max=np.array([0.005, 0.005]), phase=0, starting_pos=(-2, -2, 2, 2)):
        super().__init__(experiment=experiment, starting_position=starting_pos)
        if mating_w is None:
            mating_w = 2 * np.pi / experiment.T

        self.mating_period = np.sin(mating_w * experiment.time + phase)
        self.sigma_mating = sigma_mating
        self.mating_peak = mating_max
        self.all_mating_gradients = np.zeros((4, experiment.iterations - 1))

    def gradient_step(self, mice_position):
        m_1 = mice_position[:2]
        m_2 = mice_position[2:]

        mating_strength = -1 * 2 * (1 / self.sigma_mating ** 2) * self.mating_period[self.experiment.current_index] * \
                          (m_1 - m_2) * \
                          self.mating_energy(m_1, m_2) * \
                          self.experiment.dt

        mating_gradient = np.hstack((mating_strength, -mating_strength)).T
        self.all_mating_gradients[:, self.experiment.current_index] = mating_gradient
        self.all_gradients[:, self.experiment.current_index] = mating_gradient
        return mating_gradient

    def mating_energy(self, m_1, m_2):
        distance_squared = np.sum((m_1 - m_2) ** 2)
        return self.mating_peak * np.exp(-distance_squared / (self.sigma_mating ** 2))


default_mating_model = Mating_Model()


class Movement_Model(Model):

    def __init__(self, experiment: Experiment = default_experiment,
                 sigma_movement=np.array([0.05, 0.1]), testing=False, starting_position=np.zeros(4)):
        super().__init__(experiment=experiment, starting_position=starting_position)
        self.sigma_movement = sigma_movement

        if testing:
            np.random.seed(0)

        self.simulate_independent_movement()

    def simulate_independent_movement(self):
        extended_sigma_movement_squared = np.repeat(self.sigma_movement, 2) ** 2
        covar_scaled_with_sqrt_dt = np.diag(self.experiment.dt * extended_sigma_movement_squared)
        self.noise_driven_movement = np.random.multivariate_normal(np.zeros(4),
                                                                   covar_scaled_with_sqrt_dt,
                                                                   self.experiment.iterations).T

    def gradient_step(self, mice_position):
        gradient = self.noise_driven_movement[:, self.experiment.current_index]
        self.all_gradients[:, self.experiment.current_index] = gradient
        return gradient


default_movement_model = Movement_Model()


def sigmoid(x, half_value=100, max: float = 1, slope=25):
    return max / (1 + np.exp(- (x - half_value) / slope))

def sigmoid_2(x, half_value=np.array([100]), max: np.ndarray = np.array([1]), slope=25):
    if type(x) is np.ndarray:
        x_2_d = np.vstack((x, x))
    else:
        x_2_d = np.array([x, x])
    s = (1 + np.exp(- (x_2_d - half_value) / slope)) ** -1
    result = max * s
    if result.shape[0] == 1:
        return result[0]
    else:
        return result


class Feeding_Model(Model):

    def __init__(self, environment: Environment = default_environent, experiment: Experiment = default_experiment,
                 food_position: np.ndarray = None,
                 feeding_radius=10 ** -3,
                 hunger_strength: np.ndarray = np.array([1, 1]),
                 hunger_half_time: np.ndarray = np.array([100, 100]),
                 starting_pos=(-2, -2, 2, 2)):
        super().__init__(experiment=experiment, environment=environment, starting_position=starting_pos)

        self.food_pos = self.init_food_position(food_position)
        self.mouse_1_feed_events = list([- hunger_half_time[0]])
        self.mouse_2_feed_events = list([- hunger_half_time[1]])
        self.feeding_radius = feeding_radius
        self.hunger_freq = hunger_half_time
        self.hunger_strength = hunger_strength

    def init_food_position(self, food_position):
        if food_position is not None:
            return food_position

        return self.environment.random_vector_inside_bounds()

    def check_if_mice_are_feeding(self, mice_position):
        if np.linalg.norm(mice_position[:2] - self.food_pos) < self.feeding_radius:
            self.mouse_1_feed_events.append(self.experiment.current_time)
        if np.linalg.norm(mice_position[2:] - self.food_pos) < self.feeding_radius:
            self.mouse_2_feed_events.append(self.experiment.current_time)

    def gradient_step(self, mice_position):

        self.check_if_mice_are_feeding(mice_position)
        last_feeding = np.array([self.mouse_1_feed_events[-1], self.mouse_2_feed_events[-1]])
        next_ideal_feeding_time = last_feeding + self.hunger_freq
        mice_hunger_strength = sigmoid_2(x=self.experiment.current_time, half_value=next_ideal_feeding_time, max=self.hunger_strength)

        gradient = -1 * (mice_position - np.tile(self.food_pos, 2)) * np.repeat(mice_hunger_strength, 2) * self.experiment.dt

        self.all_gradients[:, self.experiment.current_index] = gradient
        return gradient


default_feeding_model = Feeding_Model()


class Mouse_Model(Model):

    def __init__(self, experiment: Experiment = default_experiment, starting_position=np.zeros(4),
                 environment=default_environent, movement_model: Movement_Model = default_movement_model,
                 mating_model: Mating_Model = default_mating_model,
                 feeding_model: Feeding_Model = default_feeding_model):
        super().__init__(experiment, environment, starting_position)

        self.mating_model = mating_model
        self.movement_model = movement_model
        self.feeding_model = feeding_model

    def gradient_step(self, mice_position):
        return self.movement_model.gradient_step(mice_position) + \
               self.mating_model.gradient_step(mice_position) + \
               self.feeding_model.gradient_step(mice_position)


def plot_trajectory(trajectory, fig=plt, ax=plt.gca(), color='blue', label='Mouse', show=False,
                    environment=default_environent, start_pos_color=None, show_label=True):
    if start_pos_color is None:
        start_pos_color = color
    ax.plot(trajectory[0], trajectory[1], color=color, label=f'{label} trajectory', alpha=0.6)
    ax.plot(trajectory[0, 0], trajectory[1, 0], marker='.', color=start_pos_color, markersize=25,
            label=f'{label} start position')

    if show_label:
        ax.set_xlabel('cage limits, x axis')
        ax.set_ylabel('cage limits, y axis')
    ax.set_aspect("equal")

    rect = patches.Rectangle((environment.x_lims.left - .05, environment.y_lims.left - .05),
                             environment.x_width() + 0.1,
                             environment.y_width() + 0.1, linewidth=3, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    if show:
        fig.show()
    return ax


def plot_both_mice(trajectory):
    plot_trajectory(trajectory[:2], show=False, label='Mouse 1')
    plot_trajectory(trajectory[2:], color='green', label='Mouse 2', show=True)
