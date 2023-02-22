import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from fvm import Variable2d, Equation, d1t, DT, laplass
from mesh import Quadrangle2DMesh


def boundary(mesh, arr):
    for node in mesh.nodes:
        if node.is_boundary():
            for vertex_id in node.vertexes_id:
                vertex = mesh.vertexes[vertex_id]
                if vertex.y < 2:
                    arr[node.id] = 99
                    continue
                else:
                    arr[node.id] = -99
                    continue
    return arr


if __name__ == '__main__':
    mesh = Quadrangle2DMesh(60, 60, 10, 10)
    mesh.compute()

    timesteps = 100
    time_s = 1
    k = 5
    field = Variable2d(mesh, np.zeros(len(mesh.nodes)), boundary)
    dt_var = DT(update_fn=DT.UpdatePolicies.constant_value, timesteps=timesteps, time_s=time_s)

    equation_system = [
        [d1t(field), "=", laplass(field) * k],
    ]

    equation = Equation(timesteps = timesteps)
    history = equation.evaluate([field], equation_system, dt_var)[0]

    fig, ax = plt.subplots()

    data = history[0]
    data_pic = np.zeros((mesh.x, mesh.y))
    for x_ in range(mesh.x):
        for y_ in range(mesh.y):
            data_pic[x_, y_] = data[mesh.coord_fo_idx(x_, y_)]
    sns.heatmap(data_pic, vmax=100, vmin=-100, cmap="crest")


    def animate(i):
        print(i)
        data = history[i]
        data_pic = np.zeros((mesh.x, mesh.y))
        for x_ in range(mesh.x):
            for y_ in range(mesh.y):
                data_pic[x_, y_] = data[mesh.coord_fo_idx(x_, y_)]
        sns.heatmap(data_pic, vmax=100, vmin=-100, square=True, cbar=False, cmap=sns.color_palette("vlag", as_cmap=True))


    anim = animation.FuncAnimation(fig, animate, frames=100, repeat=False, interval=10)
    plt.show()