import os
import sys

from model import default_experiment, Mating_Model, default_environent, default_mating_model

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

    ax = fig.add_subplot(2, 5, pos, projection='3d')
    ax.plot(0, 0, -1, marker='.', color='red', markersize=10)
    ax.contour3D(x, y, z, 75, cmap='Blues')
    ax.set_zorder(1)
    ax.set_xlabel('mouse 1, x component')
    ax.set_ylabel('mouse 1, y component')
    ax.set_zlabel('Energy value')

    ax.set_zlim3d(-1.01, 1.01)

    return ax

def plot_mating_energy_landscape():
    fig = plt.figure(figsize=plt.figaspect(0.5))

    examples = np.linspace(1, default_mating_model.experiment.iterations-1, 10).astype(np.int)

    ax_1 = plot_energy_3d(fig, 1, default_mating_model.mating_period[examples[0]])
    ax_2 = plot_energy_3d(fig, 2, default_mating_model.mating_period[examples[1]])
    ax_3 = plot_energy_3d(fig, 3, default_mating_model.mating_period[examples[2]])
    ax_4 = plot_energy_3d(fig, 4, default_mating_model.mating_period[examples[3]])
    ax_5 = plot_energy_3d(fig, 5, default_mating_model.mating_period[examples[4]])
    ax_6 = plot_energy_3d(fig, 6, default_mating_model.mating_period[examples[5]])
    ax_7 = plot_energy_3d(fig, 7, default_mating_model.mating_period[examples[6]])
    ax_8 = plot_energy_3d(fig, 8, default_mating_model.mating_period[examples[7]])
    ax_9 = plot_energy_3d(fig, 9, default_mating_model.mating_period[examples[8]])
    ax_10 = plot_energy_3d(fig, 10, default_mating_model.mating_period[examples[9]])
    fig.legend(labels=['Relative position of mouse 2'], markerscale=0.001, fontsize='large', loc='lower right')
    ax_3.set_title("Mating season on")
    ax_8.set_title("Mating rejection peak")
    fig.suptitle('Energy landscape of mating states', fontsize=22)
    fig.set_tight_layout(True)
    fig.show()

class Ploting_Test_Cases(unittest.TestCase):

    def test_plotting_gaussian_2_d(self):
        plot_mating_energy_landscape()