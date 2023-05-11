import json
import os

import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def mesh_variable_to_grid(var_value, Lx, Ly):
    grid = np.zeros((Lx, Ly), dtype=np.float64)
    for idx, elem in enumerate(var_value):
        x_coord = int(idx / Lx)
        y_coord = int(idx % Lx)
        grid[x_coord, y_coord] = elem
    return grid


def make_heatmap(T_history, Lx, Ly):
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.linspace(0, Lx, Lx), np.linspace(0, Ly, Ly))
    ax.pcolormesh(X, Y, T_history, vmax=100, vmin=-100)
    plt.show()


def read_var(var_path, Lx, Ly):
    var_ = []
    with open(var_path) as filee:
        for line in filee.readlines():
            var_.append(float(line[:-1]))

    return mesh_variable_to_grid(np.asarray(var_), Lx, Ly) - 273.15


if __name__ == '__main__':
    T_path = "/home/yevhen/Documents/cfdARCO/cfdARCO/dumps/T"

    Lx = 100
    Ly = 100

    T_val = read_var(T_path, Lx, Ly)
    make_heatmap(T_val, Lx, Ly)
