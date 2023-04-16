import os
import os
import sys

from matplotlib.animation import FuncAnimation

from model import Mating_Model, Mouse_Model, \
    plot_trajectory, Feeding_Model, Movement_Model

module_path = os.path.abspath(os.path.join('../src'))  # or the path to your source code
sys.path.insert(0, module_path)
import numpy as np

from matplotlib import pyplot as plt

import unittest

from plotting_utils import plot_mating_energy_landscape, \
    exemplify_mating_on_and_mating_off_trajectory_and_gradient_evolution, plot_sigmoid_function_output, \
    plot_feeding_example_trajectory, plot_example_exploration_trajectories, plot_example_full_trajectories, \
    plot_feeding_spot


class Ploting_Test_Cases(unittest.TestCase):

    def test_plotting_gaussian_2_d(self):
        plot_mating_energy_landscape()

    def test_mating_function_attraction_on(self):
        exemplify_mating_on_and_mating_off_trajectory_and_gradient_evolution()

    def test_sigmoid_implementation_default_values(self):
        plot_sigmoid_function_output()

    def test_ploting_effect_of_hunger_model_in_isolation(self):
        plot_feeding_example_trajectory()

    def test_plotting_movement_trajectories(self):
        np.random.seed(0)
        plot_example_exploration_trajectories()

    def test_plotting_full_trajectories(self):
        plot_example_full_trajectories()

    def test_other_parameters(self):
        fig, ax = plt.subplots(1, 3, layout="tight", figsize=(10, 3), sharex=True, sharey=True)
        fig.suptitle("Example trajectories of mice")

        mating_model = Mating_Model(sigma_mating=np.array([2, 2]), mating_max=np.array([1, 2]))
        feeding_model = Feeding_Model(hunger_strength=np.array([0.01, 0.02]), hunger_half_time=np.array([100, 100]))
        movement_model = Movement_Model(sigma_movement=[2, 3])

        mouse_model = Mouse_Model(starting_position=None, movement_model=movement_model,
                                  feeding_model=feeding_model,
                                  mating_model=mating_model)
        trajectory = mouse_model.simulate()
        half_experiment = int(mouse_model.experiment.iterations / 2)

        plot_trajectory(trajectory[:2], fig=fig, ax=ax[0], color='blue', label='Mouse 1',
                        start_pos_color='brown',
                        show_label=False)
        plot_trajectory(trajectory[2:], fig=fig, ax=ax[0], color='red', label='Mouse 2',
                        start_pos_color='green',
                        show_label=False)
        plot_feeding_spot(ax=ax[0], feeding_model=mouse_model.feeding_model)
        plot_trajectory(trajectory[:2, :half_experiment], fig=fig, ax=ax[1], color='blue', label='Mouse 1',
                        start_pos_color='brown',
                        show_label=False)
        plot_trajectory(trajectory[2:, :half_experiment], fig=fig, ax=ax[1], color='red', label='Mouse 2',
                        start_pos_color='green',
                        show_label=False)
        plot_feeding_spot(ax=ax[1], feeding_model=mouse_model.feeding_model)

        plot_trajectory(trajectory[:2, half_experiment:], fig=fig, ax=ax[2], color='blue', label='Mouse 1',
                        start_pos_color='brown',
                        show_label=False)
        plot_trajectory(trajectory[2:, half_experiment:], fig=fig, ax=ax[2], color='red', label='Mouse 2',
                        start_pos_color='green',
                        show_label=False)


        cols = ['Full experiment\nlength', 'Mating period on\nhalf of time', 'Rejection period on\nother half of time']
        for axis, col in zip(ax, cols):
            axis.set_title(col)


        ax[2].legend(bbox_to_anchor=[1.01, 1.02])
        fig.supxlabel('cage limits, x axis')
        fig.supylabel('cage limits, y axis')
        fig.tight_layout()
        fig.show()

    def try_animation(self):
        mating_model = Mating_Model(sigma_mating=np.array([2, 2]), mating_max=np.array([0.05, 0.05]))
        feeding_model = Feeding_Model(hunger_strength=np.array([1, 2]), hunger_half_time=np.array([100, 100]))
        movement_model = Movement_Model(sigma_movement=[0.7, 1.4])

        mouse_model = Mouse_Model(starting_position=None, movement_model=movement_model,
                                  feeding_model=feeding_model,
                                  mating_model=mating_model)
        trajectory = mouse_model.simulate()

        fig = plt.figure()
        ax = plt.axes(xlim=(-4, 4), ylim=(-4, 4))
        line, = ax.plot([], [], lw=2)

        # initialization function
        def init():
            # creating an empty plot/frame
            line.set_data([], [])
            return line,

            # lists to store x and y axis points

        xdata, ydata = [], []

        # animation function
        def animate(i):
            # t is a parameter
            t = 2 * i

            # x, y values to be plotted
            x = trajectory[0, i]
            y = trajectory[1, i]

            # appending new points to x, y axes points list
            xdata.append(x)
            ydata.append(y)
            line.set_data(xdata, ydata)
            return line,

            # setting a title for the plot

        plt.title('Creating a growing coil with matplotlib!')
        # hiding the axis details
        plt.axis('off')

        # call the animator
        anim = FuncAnimation(fig, animate, init_func=init,
                                       frames=500, interval=20, blit=True)

        # save the animation as mp4 video file
        anim.save('coil.gif', writer='imagemagick')

    def test_celluloid(self):

        from celluloid import Camera
        mating_model = Mating_Model(sigma_mating=np.array([1, 1]), mating_max=np.array([8, 8]))
        feeding_model = Feeding_Model(hunger_strength=np.array([1, 2]), hunger_half_time=np.array([100, 100]))
        movement_model = Movement_Model(sigma_movement=[0.07, 0.09])

        mouse_model = Mouse_Model(starting_position=None, movement_model=movement_model,
                                  feeding_model=feeding_model,
                                  mating_model=mating_model)
        trajectory = mouse_model.simulate()

        fig, ax = plt.subplots(1, 1)
        camera = Camera(fig)
        for i in range(100):
            plot_trajectory(trajectory[:2, i*100:(i+1)*100], fig=fig, ax=ax, color='blue', label='Mouse 1',
                            start_pos_color='brown',
                            show_label=False)
            plot_trajectory(trajectory[2:, i*100:(i+1)*100], fig=fig, ax=ax, color='red', label='Mouse 2',
                            start_pos_color='green',
                            show_label=False)
            plot_feeding_spot(ax=ax, feeding_model=mouse_model.feeding_model)
            camera.snap()


        #self.plot_gradients(ax, mouse_model, movement_model, label='Movement')
        #self.plot_gradients(ax, mouse_model, feeding_model, label='Feeding')
        #self.plot_gradients(ax, mouse_model, mating_model, label='Mating')

        animation = camera.animate()
        animation.save('movement.gif', writer='imagemagick')
        #fig.legend()
        fig.show()

    def plot_gradients(self, ax, mouse_model, model, label):
        grad_val_m1 = self.grad_value(model.all_gradients[:2])
        ax[1].plot(mouse_model.experiment.time[:500],
                   grad_val_m1[:500],
                   label=f'{label} Gradients mouse 1', alpha=0.5)
        print(f"{label} - Mouse 1  Min {np.min(grad_val_m1)}, "
              f"Max  {np.max(grad_val_m1)}, "
              f"Average  {np.average(grad_val_m1)}")
        grad_val_m2 = self.grad_value(model.all_gradients[2:])
        ax[1].plot(mouse_model.experiment.time[:500],
                   grad_val_m2[:500],
                   label=f'{label} Gradients mouse 2', alpha=0.5)
        print(f"{label} - Mouse 2  Min {np.min(grad_val_m2)}, "
              f"Max  {np.max(grad_val_m2)}, "
              f"Average  {np.average(grad_val_m2)}")

    def grad_value(self, array):
        return np.sum(array ** 2, axis=0)


if __name__ == '__main__':
    unittest.main()
