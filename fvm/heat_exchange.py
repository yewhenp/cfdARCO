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
                if vertex.x < 0.9:
                    arr[node.id] = 99
                    continue
                elif vertex.x > 8.9:
                    arr[node.id] = -99
                    continue
                else:
                    arr[node.id] = 0
                    continue
    return arr


if __name__ == '__main__':
    nX = 20
    nY = 20
    lX = 10
    lY = 10
    mesh = Quadrangle2DMesh(nX, nY, lX, lY)
    last_dist = 2
    for x in range(nX):
        node_id = mesh.coord_fo_idx(x, 0)
        vrtx_id = mesh.vertexes[mesh.nodes[node_id].vertexes_id[1]].id
        mesh.vertexes[vrtx_id].coords[1] += (x + 1) * last_dist
        for y in range(nY):
            node_id = mesh.coord_fo_idx(x, y)
            vrtx_id = mesh.vertexes[mesh.nodes[node_id].vertexes_id[2]].id
            mesh.vertexes[vrtx_id].coords[1] += (x+1) * last_dist
            last_dist -= (1 / (nX + nY)) / (x+1)

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
    equation.evaluate([field], equation_system, dt_var)

    fig, ax = plt.subplots()
    xv, yv = mesh.get_meshgrid()

    def animate(i):
        print(i)
        data = field.history[i]
        data_pic = np.zeros((mesh.x, mesh.y))
        for x_ in range(mesh.x):
            for y_ in range(mesh.y):
                data_pic[x_, y_] = data[mesh.coord_fo_idx(x_, y_)]
        ax.pcolormesh(xv, yv, data_pic, vmax=100, vmin=-100, cmap=sns.color_palette("vlag", as_cmap=True))


    anim = animation.FuncAnimation(fig, animate, frames=timesteps, repeat=False, interval=10)
    plt.show()