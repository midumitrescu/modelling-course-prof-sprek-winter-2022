import unittest

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import *

from model import Feeding_Model, Mouse_Model, Movement_Model, Experiment, Mating_Model, plot_trajectory, Environment, \
    Box_limits, sigmoid, plot_both_mice, sigmoid_2

default_experiment = Experiment()
default_mating_model = Mating_Model()

default_environment = Environment()
no_feedind = type("No_Feeding_Model", (Feeding_Model, object), {"gradient": lambda self, _: np.zeros(4)})()
no_movement = type("No_Movement_Model", (Movement_Model, object), {"gradient": lambda self, _: np.zeros(4)})()


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
        very_low_movement = Movement_Model(sigma_movement=np.array([10 ** -5, 10 ** -5]),
                                           starting_position=(-2, -2, 2, 2))
        trajectory = very_low_movement.simulate()

        np.apply_along_axis(lambda array: assert_array_almost_equal([-2, -2, 2, 2], array, decimal=4,
                                                                    err_msg='Mice should not move with such low sigma'),
                            arr=trajectory, axis=0)

        plot_both_mice(trajectory)

    def test_movement_model(self):
        np.random.seed(0)
        somewhat_high_movement_model = Movement_Model(sigma_movement=np.array([0.1, 0.2]),
                                                      starting_position=(np.array([-2, -2, 2, 2])))
        trajectory = somewhat_high_movement_model.simulate()

        #assert_array_almost_equal(np.array([
        #    [-2.00000, -1.99690, -1.99390, -1.99345, -1.99204, -1.99105, -1.98832, -1.98817, -1.98768, -1.98878],
        #    [-2.00000, -1.99291, -1.99339, -1.98879, -1.98774, -1.99044, -1.99279, -1.99338, -1.99218, -1.99169],
        #    [2.00000, 2.01116, 2.02297, 2.02232, 2.02713, 2.03658, 2.02043, 2.03479, 2.04448, 2.03887],
        #    [2.00000, 2.00253, 1.99635, 1.99895, 1.99972, 1.99842, 2.00255, 1.99335, 2.00265, 1.99012]]),
        #    trajectory[:, :10], decimal=5)

    def test_movement_variance(self):
        np.random.seed(0)

        stds = [(0.1, 0.2), (0.5, 0.5), (1, 1)]

        for std in stds:
            somewhat_high_movement_model = Movement_Model(sigma_movement=np.array(std))
            somewhat_high_movement_model.simulate_independent_movement()
            movement = somewhat_high_movement_model.noise_driven_movement
            assert_array_almost_equal([std[0] ** 2 * default_experiment.dt,
                                       std[0] ** 2 * default_experiment.dt,
                                       std[1] ** 2 * default_experiment.dt,
                                       std[1] ** 2 * default_experiment.dt],
                                      np.var(movement, axis=1), decimal=1, err_msg=f"Variance for {std} is wrong")


class Mating_Model_Test_Cases(unittest.TestCase):

    def test_mating_function_repulsion_on(self):
        never_mating = Mating_Model(sigma_mating=np.array([2.5, 2.5]),
                                    mating_max=np.array([0.005, 0.005]))
        never_mating.mating_period = -1 * np.ones(default_experiment.time.shape)
        never_mating.mice_trajectory[:, 0] = [-0.05, -0.05, 1, 0.5]

        trajectory = never_mating.simulate()

        plot_trajectory(trajectory[:2], show=False, label='Mouse 1')
        plot_trajectory(trajectory[2:], color='green', label='Mouse 2', show=True)

        fig, ax = plt.subplots(1, 1)
        ax.plot(default_experiment.time, np.linalg.norm(never_mating.all_gradients[:2, :], axis=0),
                label='Mating desire of Mouse 1')
        ax.plot(default_experiment.time, np.linalg.norm(never_mating.all_gradients[2:, :], axis=0),
                label='Mating desire of Mouse 2')
        fig.legend()
        fig.show()

        plot_both_mice(trajectory)

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
        feeding_model = Feeding_Model(food_position=np.array([0, 0]), hunger_strength=np.array([0.1, 0.2]))
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


class Sigmoid_Test_Cases(unittest.TestCase):

    def test_sigmoid_test_case_easy(self):
        assert_array_almost_equal([0.5, 1], sigmoid(0, half_value=np.array([0, 0]), max = np.array([1, 2])))

    def test_sigmoid_implementation_1(self):
        x = np.linspace(0, 200, 400)
        z = sigmoid(x, half_value=50, max=np.array([50]))
        self.assertAlmostEqual(25, z[100], places=0)

        plt.plot(x, z)
        plt.xlabel("x")
        plt.ylabel("Sigmoid(X)")
        plt.show()

    def test_sigmoid_implementation_2(self):
        x = np.linspace(-100, 300, 400)
        max = np.array([50])
        z_default = sigmoid(x, half_value=100, max=max, slope=10)
        self.assertTrue(z_default[199] < 25 < z_default[200])
        z_steep = sigmoid(x, half_value=100, max=max, slope=5)
        z_flat = sigmoid(x, half_value=100, max=max, slope=50)

        plt.plot(x, z_default, color='red', label='Default slope')
        plt.plot(x, z_steep, color='blue', label='Steep slope')
        plt.plot(x, z_flat, color='green', label='Flat slope')
        plt.xlabel("x")
        plt.ylabel("Sigmoid(X)")
        plt.legend()
        plt.show()

    def test_sigmoid_accepts_array_as_input(self):
        x = np.linspace(-100, 300, 400)
        z = sigmoid_2(x, half_value=np.array([100, 100]), max=np.array([50, 100]))
        assert_array_almost_equal([25, 50], z[:, 200], decimal=0)

        plt.plot(x, z[0])
        plt.plot(x, z[1])
        plt.xlabel("x")
        plt.ylabel("Sigmoid(X)")
        plt.show()

    def test_sigmoid_accepts_two_array_as_input(self):
        x = np.linspace(-100, 300, 400)
        z = sigmoid_2(x, half_value=np.array([0, 100]), max=np.array([50, 100]))

        plt.plot(x, z[0], label='half value 0, max 50')
        plt.plot(x, z[1], label='half value 100, max 100')
        plt.xlabel("x")
        plt.ylabel("Sigmoid(X)")
        plt.legend()
        plt.show()

    def test_half_value_1(self):
        assert_array_almost_equal([25, 1.798621], sigmoid_2(0, half_value=np.array([0, 100]), max=np.array([50, 100])))

    def test_half_value_2(self):
        assert_array_almost_equal([49.10069, 50], sigmoid_2(100, half_value=np.array([0, 100]), max=np.array([50, 100])))

    def test_call_from_model(self):
        x = 0.0
        feeding_time = np.array([0, 0])
        hunger_strength = np.array([1, 1])

        result = sigmoid(x, feeding_time, hunger_strength)

        assert_array_almost_equal([0.5, 0.5], result)
        self.assertEqual((2,), result.shape)


if __name__ == '__main__':
    unittest.main()
