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


def make_streamplot(u_history, v_history, Lx, Ly):
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.linspace(0, Lx, Lx), np.linspace(0, Ly, Ly))

    # def animate(i):
    #     print(i)
    #     u_curr_history = u_history[i]
    #     v_curr_history = v_history[i]
    #     ax.clear()
    #     ax.streamplot(X, Y, u_curr_history, v_curr_history)
    #
    # anim = animation.FuncAnimation(fig, animate, frames=len(u_history), repeat=False)

    u_curr_history = u_history[-1]
    v_curr_history = v_history[-1]
    ax.streamplot(X, Y, u_curr_history, v_curr_history)
    plt.show()


def read_var(var_path, Lx, Ly):
    u_var_history = []
    v_var_history = []

    with open(var_path) as filee:
        for line in filee.readlines():
            if line.startswith("(") and len(line) > 3:
                u_str, v_str = line.split(" ")[0][1:], line.split(" ")[1]
                u_var_history.append(float(u_str))
                v_var_history.append(float(v_str))

    return [mesh_variable_to_grid(np.asarray(u_var_history), Lx, Ly)], [mesh_variable_to_grid(np.asarray(v_var_history), Lx, Ly)]


if __name__ == '__main__':
    x = 20
    y = 20

    u_history, v_history = read_var("../dumps/U", x, y)
    make_streamplot(u_history, v_history, x, y)
