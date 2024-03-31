import json
import os
import time

import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np


def mesh_variable_to_grid(var_value, Lx, Ly):
    grid = np.asarray(var_value, dtype=np.float64)
    grid = grid.reshape((Lx, Ly))
    # grid = np.zeros((Lx, Ly), dtype=np.float64)
    # for idx, elem in enumerate(var_value):
    #     x_coord = int(idx % Lx)
    #     y_coord = int(idx / Lx)
    #     grid[x_coord, y_coord] = elem
    return grid


def make_heatmap(T_history, Lx, Ly, name):
    min_elem = T_history[0].min()
    max_elem = T_history[0].max()
    pre_min_elem = 0
    pre_max_elem = 0

    # for el in T_history:
    #     curr_min_elem = el.min()
    #     curr_max_elem = el.max()
    #     if curr_min_elem < min_elem:
    #         pre_min_elem = min_elem
    #         min_elem = curr_min_elem
    #     if curr_max_elem > max_elem:
    #         pre_max_elem = max_elem
    #         max_elem = curr_max_elem

    for el in T_history:
        min_elem = min(min_elem, el.min())
        max_elem = max(max_elem, el.max())

    r_max = max_elem
    r_min = min_elem
    # r_max = 0.03
    # r_min = -0.03

    if len(T_history) > 1:
        print("Animating")
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # fig, ax = plt.subplots()
        X, Y = np.meshgrid(np.linspace(0, Lx, Lx), np.linspace(0, Ly, Ly))
        def animate(i):
            print(i)
            data = T_history[i]
            ax.cla()
            ax.plot_surface(X, Y, data, cmap=cm.coolwarm)
            # ax.pcolormesh(X, Y, data, vmax=r_max, vmin=r_min)
            # ax.pcolormesh(X, Y, data)
            ax.set_zlim(r_min, r_max)
        anim = animation.FuncAnimation(fig, animate, frames=len(T_history), repeat=False, interval=1)
        writer = animation.PillowWriter(fps=60,
                                    metadata=dict(artist='Me'),
                                    bitrate=1800)
        # anim.save(f'{name}.gif', writer=writer)
    else:
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(np.linspace(0, Lx, Lx), np.linspace(0, Ly, Ly))
        print("Last show")
        # ax.pcolormesh(X, Y, T_history[0], vmax=100, vmin=-5)
        ax.pcolormesh(X, Y, T_history[0])
        plt.axis('off')
        plt.savefig('mesh_sim.pdf')
    plt.show()

    # pygame.init()
    # display = pygame.display.set_mode((Lx, Ly))
    # pygame.display.set_caption("Solving the 2d Wave Equation")
    #
    #
    # i = 0
    # while True:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #             return
    #
    #     pixeldata = np.zeros((Lx, Ly, 3), dtype=np.uint8)
    #     pixeldata[:, :, 0] = np.clip(T_history[i] + 128, 0, 255)
    #     pixeldata[:, :, 1] = 0
    #     pixeldata[:, :, 2] = 0
    #
    #     surf = pygame.surfarray.make_surface(pixeldata)
    #     display.blit(pygame.transform.scale(surf, (Lx, Ly)), (0, 0))
    #     pygame.display.update()
    #
    #     i+=1
    #     time.sleep(0.1)


def read_var(var_path, Lx, Ly, use_last_only):
    var_history = []
    filepathes = []

    for filename in tqdm.tqdm(os.listdir(var_path)):
        f = os.path.join(var_path, filename)
        if os.path.isfile(f):
            filepathes.append(f)

    stepp = 20

    if not use_last_only:
        for i in tqdm.trange(len(filepathes[::stepp])):
            var = np.fromfile(var_path + "/" + str(i * stepp) + ".bin", dtype="float64")
            var_history.append(mesh_variable_to_grid(var, Lx, Ly))
    else:
        var = np.fromfile(var_path + "/" + str(len(filepathes) - 1) + ".bin", dtype="float64")
        var_history.append(mesh_variable_to_grid(var, Lx, Ly))

    return var_history


if __name__ == '__main__':
    base_dir = "../dumps/run_latest/"

    with open(base_dir + "/mesh.json") as filee:
        mesh_json = json.load(filee)
    mesh = []

    for node in mesh_json["nodes"]:
        node_repr = []
        for v_id in node["vertexes"]:
            node_repr.append(mesh_json["vertexes"][v_id])
        mesh.append(node_repr)


    use_last_only = 0

    rho = read_var(base_dir + "/h_shallow/", mesh_json["x"], mesh_json["y"], use_last_only)
    print(mesh_json.keys())
    for i in range(len(rho)):
        rho[i] = rho[i] - 100
    make_heatmap(rho, mesh_json["x"], mesh_json["y"], "h_shallow")