import os
import sys

from model import default_experiment, Mating_Model, default_environent, default_mating_model, Mouse_Model, \
    plot_trajectory, sigmoid, Feeding_Model, Movement_Model

module_path = os.path.abspath(os.path.join('../src'))  # or the path to your source code
sys.path.insert(0, module_path)
import numpy as np

from matplotlib import pyplot as plt


def plot_energy_3d(fig, pos, max, origin=(0, 0), sigma=2):
    x = default_environent.x
    y = default_environent.y

    x, y = np.meshgrid(x, y)

    z = -1 * max * np.exp(- ((x - origin[0]) ** 2 + (y - origin[1]) ** 2) / sigma ** 2)

    ax = fig.add_subplot(1, 3, pos, projection='3d')
    ax.plot(0, 0, -1, marker='.', color='red', markersize=25)
    ax.set_zorder(1)
    ax.contour3D(x, y, z, 30, cmap='Blues', alpha=0.7)
    ax.set_xlabel('mouse 1, x component')
    ax.set_ylabel('mouse 1, y component')
    ax.set_zlabel('Energy value')

    ax.set_zlim3d(-1.05, 1.05)

    return ax


def plot_mating_energy_landscape(figsize=(20, 10)):
    fig = plt.figure(figsize=figsize)

    ax_1 = plot_energy_3d(fig, 1, -1)
    ax_2 = plot_energy_3d(fig, 2, default_mating_model.mating_period[1])
    ax_3 = plot_energy_3d(fig, 3, 1)
    fig.legend(labels=['Relative position of mouse 2'], markerscale=0.001, fontsize='large', loc='lower right')
    ax_1.set_title("Mice rejecting each other")
    ax_2.set_title("Mice are neutral \n to mating")
    ax_3.set_title("Mating season on")
    fig.suptitle('Energy landscape of mating states', fontsize=22)
    fig.set_tight_layout(False)
    fig.show()


def plot_mating_trajectory(model: Mating_Model, fig, ax,
                           ax_title='when mating season is peaking'):
    ax_m1 = plot_trajectory(model.mice_trajectory[:2], fig, ax, label='Mouse 1')
    ax_m2 = plot_trajectory(model.mice_trajectory[2:], fig, ax, color='red', label='Mouse 2')
    ax_m1.plot()
    ax.set_title(ax_title)


def exemplify_mating_on_and_mating_off_trajectory_and_gradient_evolution(figsize=(9, 9)):
    always_mating, always_rejecting = plot_mice_trajetories_for_mating_season_on_and_off(figsize)
    plot_mating_desire_gradients_for_mating_on_and_off(always_mating, always_rejecting, figsize)


def plot_mating_desire_gradients_for_mating_on_and_off(always_mating, always_rejecting, figsize=(10, 4)):
    fig, ax = plt.subplots(1, 2, layout="tight", figsize=figsize, sharex=True, sharey=True)
    fig.suptitle("Dynamical evolution of mating desire gradient size")
    plot_mating_desire(always_mating, fig, ax[0], ax_title='when mating season is peaking')
    plot_mating_desire(always_rejecting, fig, ax[1], ax_title='when mice reject each other')
    ax[1].legend(loc='center right')
    fig.tight_layout()
    fig.show()


def plot_mice_trajetories_for_mating_season_on_and_off(figsize=(9, 9)):
    fig, ax = plt.subplots(1, 2, layout="tight", figsize=figsize)
    fig.suptitle("Example of mice trajectories")
    always_mating = simulate_always_mating()
    always_rejecting = simulate_always_rejecting()
    plot_mating_trajectory(always_mating, fig, ax[0], ax_title='when mating season is peaking')
    plot_mating_trajectory(always_rejecting, fig, ax[1], ax_title='when mice reject each other')
    ax[1].legend(loc='lower right')
    fig.show()
    return always_mating, always_rejecting


def simulate_always_mating():
    always_mating = Mating_Model(sigma_mating=np.array([2.5, 2.5]),
                                 mating_max=np.array([0.05, 0.05]))
    always_mating.mating_period = np.ones(default_experiment.time.shape)

    always_mating.simulate()
    return always_mating


def simulate_always_rejecting():
    always_rejecting = Mating_Model(sigma_mating=np.array([2.5, 2.5]),
                                    mating_max=np.array([0.05, 0.05]),
                                    starting_pos=[-0.5, -0.5, 0.5, 0.5])
    always_rejecting.mating_period = -1 * np.ones(default_experiment.time.shape)

    always_rejecting.simulate()
    return always_rejecting


def plot_mating_desire(mating_model, fig, ax, ax_title):
    ax.plot(default_experiment.time[1:], np.linalg.norm(mating_model.all_mating_gradients[:2, :], axis=0),
            label='Mating desire of Mouse 1')
    ax.plot(default_experiment.time[1:], np.linalg.norm(mating_model.all_mating_gradients[2:, :], axis=0),
            label='Mating desire of Mouse 2', color='red')
    ax.set_xlabel('iterations')
    ax.set_ylabel('mating intensity')
    ax.set_title(ax_title)


def plot_sigmoid_function_output():
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    x = np.linspace(0, 300, 300)
    z = sigmoid(x, half_value=150)
    ax.plot(x, z, label='σ(t)')
    ax.plot(0, 0, marker='^', color='red', markersize=10, label='not hungry at all')
    ax.plot(150, 0, marker='^', color='green', markersize=10, label='quite hungry')
    ax.plot(300, 0, marker='^', color='black', markersize=10, label='extremely hungry')
    ax.plot(x, sigmoid(x, half_value=150, slope=35), label='mouse gets hungry slower')
    ax.plot(x, sigmoid(x, half_value=150, slope=10), label='mouse gets hungry quicker')
    ax.hlines(y=0.5, xmin=0, xmax=150, colors='black', linestyles='dotted', label='σ(t) = 0.5')
    ax.vlines(x=150, ymin=0, ymax=0.5, colors='black', linestyles='dotted')
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("σ(t)")
    ax.legend(bbox_to_anchor=(1.01, 1.01))
    fig.suptitle(
        'Sigmoid value evolution in time with respect to last feeding time. \n \n Example is: mice has eaten at '
        '\n t=0 and he gets quite '
        'hungry in 150 ms')
    fig.tight_layout()
    fig.show()


def plot_feeding_example_trajectory(figsize=(4.5, 4.5)):
    feeding_model = Feeding_Model(food_position=np.array([0, 0]), starting_pos=[-2, -2, 1, 2.7])

    fig, ax = plt.subplots(1, 1, layout="tight", figsize=figsize)
    fig.suptitle("Example of mice trajectories \n driven by hunger")

    trajectory = feeding_model.simulate()
    plot_trajectory(trajectory[:2], fig=fig, ax=ax, color='blue', label='Mouse 1')
    plot_trajectory(trajectory[2:], fig=fig, ax=ax, color='red', label='Mouse 2')

    plot_feeding_spot(ax, feeding_model)
    ax.legend(loc='lower right')
    fig.show()


def plot_feeding_spot(ax, feeding_model):
    ax.plot(feeding_model.food_pos[0], feeding_model.food_pos[1], marker='.', color='yellow', markersize=25,
            label=f'Food position')


def plot_example_exploration_trajectories():
    fig, ax = plt.subplots(2, 3, layout="tight", figsize=(13, 6), sharex=True, sharey=True)
    fig.suptitle("Example trajectories of mice \n exploring using gaussian white noise")
    for axis in ax.flatten():
        example_movement = Movement_Model(starting_position=None, testing=False,
                                          sigma_movement=np.array([0.07, 0.14]))
        trajectory = example_movement.simulate()
        plot_trajectory(trajectory[:2], fig=fig, ax=axis, color='blue', label='Mouse 1', start_pos_color='brown',
                        show_label=False)
        plot_trajectory(trajectory[2:], fig=fig, ax=axis, color='red', label='Mouse 2', start_pos_color='green',
                        show_label=False)
    ax[1, 2].legend(bbox_to_anchor=[1.01, 1.02])
    fig.supxlabel('cage limits, x axis')
    fig.supylabel('cage limits, y axis')
    fig.tight_layout()
    fig.show()


def plot_example_full_trajectories(figsize=(10, 6)):
    fig, ax = plt.subplots(3, 3, layout="tight", figsize=figsize, sharex=True, sharey=True)
    fig.suptitle("Example trajectories of mice")

    for index in range(3):
        mating_model = Mating_Model(sigma_mating=np.array([3, 3]), mating_max=np.array([0.1, 0.1]))
        feeding_model = Feeding_Model(hunger_strength=np.array([0.01, 0.02]), hunger_half_time=np.array([100, 100]))
        movement_model = Movement_Model(sigma_movement=[0.2, 0.3])
        mouse_model = Mouse_Model(starting_position=None, movement_model=movement_model, feeding_model=feeding_model, mating_model=mating_model)
        trajectory = mouse_model.simulate()
        half_experiment = int(mouse_model.experiment.iterations / 2)

        plot_trajectory(trajectory[:2], fig=fig, ax=ax[index, 0], color='blue', label='Mouse 1',
                        start_pos_color='brown',
                        show_label=False)
        plot_trajectory(trajectory[2:], fig=fig, ax=ax[index, 0], color='red', label='Mouse 2', start_pos_color='green',
                        show_label=False)
        plot_feeding_spot(ax=ax[index, 0], feeding_model=mouse_model.feeding_model)

        plot_trajectory(trajectory[:2, :half_experiment], fig=fig, ax=ax[index, 1], color='blue', label='Mouse 1',
                        start_pos_color='brown',
                        show_label=False)
        plot_trajectory(trajectory[2:, :half_experiment], fig=fig, ax=ax[index, 1], color='red', label='Mouse 2',
                        start_pos_color='green',
                        show_label=False)
        plot_feeding_spot(ax=ax[index, 1], feeding_model=mouse_model.feeding_model)

        plot_trajectory(trajectory[:2, half_experiment:], fig=fig, ax=ax[index, 2], color='blue', label='Mouse 1',
                        start_pos_color='brown',
                        show_label=False)
        plot_trajectory(trajectory[2:, half_experiment:], fig=fig, ax=ax[index, 2], color='red', label='Mouse 2',
                        start_pos_color='green',
                        show_label=False)
        plot_feeding_spot(ax=ax[index, 2], feeding_model=mouse_model.feeding_model)
    cols = ['Full experiment\nlength', 'Mating period on\nhalf of time', 'Rejection period on\nother half of time']
    for axis, col in zip(ax[0], cols):
        axis.set_title(col)

    for axis, row in zip(ax[:, 0], range(3)):
        axis.set_ylabel(f'Simulation\n#{row+1}', rotation=0, size='large')

    ax[1, 2].legend(bbox_to_anchor=[1.01, 1.02])
    fig.supxlabel('cage limits, x axis')
    fig.supylabel('cage limits, y axis')
    fig.tight_layout()
    fig.show()










