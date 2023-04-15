import os
import sys

from model import default_experiment, Mating_Model, default_environent, default_mating_model, Mouse_Model, \
    plot_trajectory

module_path = os.path.abspath(os.path.join('../src')) # or the path to your source code
sys.path.insert(0, module_path)
import numpy as np
import pdb

from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

import unittest


def plot_energy_3d(fig, pos, max, origin=(0, 0), sigma=2):

    x = default_environent.x
    y = default_environent.y

    x, y = np.meshgrid(x, y)

    z = max * np.exp(- ((x - origin[0]) ** 2 + (y - origin[1]) ** 2) / sigma**2)

    ax = fig.add_subplot(1, 3, pos, projection='3d')
    ax.plot(0, 0, -1, marker='.', color='red', markersize=25)
    ax.set_zorder(1)
    ax.contour3D(x, y, z, 30, cmap='Blues')
    ax.set_xlabel('mouse 1, x component')
    ax.set_ylabel('mouse 1, y component')
    ax.set_zlabel('Energy value')

    ax.set_zlim3d(-1.05, 1.05)

    return ax

def plot_mating_energy_landscape(figsize=(20, 10)):
    fig = plt.figure(figsize=figsize)

    ax_1 = plot_energy_3d(fig, 1, np.min(default_mating_model.mating_period))
    ax_2 = plot_energy_3d(fig, 2, default_mating_model.mating_period[1])
    ax_3 = plot_energy_3d(fig, 3, np.max(default_mating_model.mating_period))
    fig.legend(labels=['Relative position of mouse 2'], markerscale=0.001, fontsize='large', loc='lower right')
    ax_1.set_title("Mice rejecting each other")
    ax_2.set_title("Mice are neutral \n to mating")
    ax_3.set_title("Mating season on")
    fig.suptitle('Energy landscape of mating states', fontsize=22)
    fig.set_tight_layout(False)
    fig.show()

def simulate_and_plot_always_mating_scenario():
    fig, ax = plt.subplots(1, 1, layout="tight")
    always_mating = simulate_always_mating()

    ax_m1 = plot_trajectory(always_mating.mice_trajectory[:2], fig, ax, show=False, label='Mouse 1 trajectory')
    ax_m2 = plot_trajectory(always_mating.mice_trajectory[2:], fig, ax, color='red', label='Mouse 2 trajectory', show=False)
    ax_m1.plot(-2, -2, marker='.', color='blue', markersize=25, label='Starting position of mouse 1')
    ax_m2.plot(2, 2, marker='.', color='red', markersize=25, label='Starting position of mouse 2')
    fig.legend(loc='center right')
    fig.suptitle('Example of mice trajectories \n when mating season is peaking')
    fig.show()
    return always_mating


def simulate_always_mating():
    always_mating = Mating_Model(sigma_mating=np.array([2.5, 2.5]),
                                 mating_peak=np.array([0.005, 0.005]))
    always_mating.mating_period = np.ones(default_experiment.time.shape)

    always_mating.simulate()
    return always_mating


def plot_mating_desire(mating_model):
    fig, ax = plt.subplots(1, 1, layout="tight")

    ax.plot(default_experiment.time[1:], np.linalg.norm(mating_model.all_mating_gradients[:2, :], axis=0),
            label='Mating desire of Mouse 1')
    ax.plot(default_experiment.time[1:], np.linalg.norm(mating_model.all_mating_gradients[2:, :], axis=0),
            label='Mating desire of Mouse 2', color='red')
    ax.set_xlabel('iterations')
    ax.set_ylabel('mating intensity')
    fig.suptitle('Dynamical evolution of mating desire gradient size')
    fig.legend(loc='center right')
    fig.show()

class Ploting_Test_Cases(unittest.TestCase):

    def test_plotting_gaussian_2_d(self):
        plot_mating_energy_landscape()

    def test_mating_function_attraction_on(self):
        always_mating = simulate_and_plot_always_mating_scenario()
        plot_mating_desire(always_mating)