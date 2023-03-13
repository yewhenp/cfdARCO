import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fvm import Variable2d, Equation, d1dx, d1dy, d1t, DT, laplass
from mesh import Quadrangle2DMesh
from matplotlib import cm


def boundary_none(mesh, arr):
    return arr



def initial_pertrbations(mesh):
    arr = np.zeros(len(mesh.nodes))
    for node in mesh.nodes:
        if .30 < node.x < .70 and .30 < node.y < .70:
            arr[node.id] = 1.3
        else:
            arr[node.id] = 1
    return arr



if __name__ == '__main__':
    mesh = Quadrangle2DMesh(100, 100, 1, 1)
    mesh.compute()

    timesteps = 100
    CFL = 0.8
    g = 9.81

    h = Variable2d(mesh, initial_pertrbations(mesh), boundary_none, "rho")
    u = Variable2d(mesh, np.zeros(len(mesh.nodes)), boundary_none, "u")
    v = Variable2d(mesh, np.zeros(len(mesh.nodes)), boundary_none, "v")
    h_u = Variable2d(mesh, np.zeros(len(mesh.nodes)), boundary_none, "rho_u")
    h_v = Variable2d(mesh, np.zeros(len(mesh.nodes)), boundary_none, "rho_v")
    dt = DT(update_fn=DT.UpdatePolicies.CourantFriedrichsLewy, CFL=CFL, space_vars=[u, v])


    equation_system = [
        [d1t(h), "=", -(d1dx(h_u) + d1dy(h_v))],
        [d1t(h_u),    "=", -(d1dx(h_u*h_u/h + 0.5*g*h*h)  + d1dy(h_v*h_u/h)) ],
        [d1t(h_v),    "=", -(d1dx(h_v*h_u/h)  + d1dy(h_v*h_v/h + 0.5*g*h*h)) ],
    ]


    equation = Equation(timesteps)
    equation.evaluate([h, h_u, h_v], equation_system, dt)


    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = np.arange(0, 100, 1)
    Y = np.arange(0, 100, 1)
    X, Y = np.meshgrid(X, Y)
    def animate(i):
        print(i)

        h_pic = np.zeros((mesh.x, mesh.y))
        data = h.history[i]
        for x_ in range(mesh.x):
            for y_ in range(mesh.y):
                h_pic[x_, y_] = data[mesh.coord_fo_idx(x_, y_)]

        ax.clear()
        surf = (ax.plot_surface(X, Y, h_pic, cmap=cm.coolwarm, vmin=-1, vmax=3))
        ax.set_zlim(-1.01, 1.01)

        return surf

    anim = animation.FuncAnimation(fig, animate, frames=timesteps)
    anim.save("cfd.gif")
    plt.show()
