import json
import os

import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def mesh_variable_to_grid(var_value, Lx, Ly):
    grid = np.zeros((Lx, Ly), dtype=np.float64)
    for idx, elem in enumerate(var_value):
        x_coord = int(idx % Lx)
        y_coord = int(idx / Lx)
        grid[x_coord, y_coord] = elem
    return grid


def make_heatmap(T_history, Lx, Ly):
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.linspace(0, Lx, Lx), np.linspace(0, Ly, Ly))

    if len(T_history) > 1:
        print("Animating")

        def animate(i):
            print(i)
            data = T_history[i]
            ax.pcolormesh(X, Y, data, vmax=100, vmin=-100)

        anim = animation.FuncAnimation(fig, animate, frames=len(T_history), repeat=False, interval=10)
    else:
        print("Last show")
        ax.pcolormesh(X, Y, T_history[0], vmax=100, vmin=-100)
    plt.show()


def read_var(var_path, Lx, Ly):
    var_history = []
    filepathes = []
    for filename in tqdm.tqdm(os.listdir(var_path)):
        f = os.path.join(var_path, filename)
        if os.path.isfile(f):
            filepathes.append(f)

    for i in range(len(filepathes)):
        var = np.fromfile(var_path + "/" + str(i) + ".bin", dtype="float64")
        var_history.append(mesh_variable_to_grid(var, Lx, Ly))

    return var_history


if __name__ == '__main__':
    base_dir = "/home/yevhen/Documents/cfdARCO/cfdARCO/dumps/run_latest/"

    with open(base_dir + "/mesh.json") as filee:
        mesh_json = json.load(filee)
    mesh = []

    for node in mesh_json["nodes"]:
        node_repr = []
        for v_id in node["vertexes"]:
            node_repr.append(mesh_json["vertexes"][v_id])
        mesh.append(node_repr)

    T_history = read_var(base_dir + "/T/", mesh_json["x"], mesh_json["y"])
    make_heatmap(T_history, mesh_json["x"], mesh_json["y"])
