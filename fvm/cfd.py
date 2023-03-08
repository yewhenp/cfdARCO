import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fvm import Variable2d, Equation, d1dx, d1dy, d1t, DT, laplass
from mesh import Quadrangle2DMesh


def boundary_u(mesh, arr):
    for node in mesh.nodes:
        if node.is_boundary():
            for vertex_id in node.vertexes_id:
                vertex = mesh.vertexes[vertex_id]
                if vertex.y < 1.5:
                    arr[node.id] = 1
                    continue
                else:
                    arr[node.id] = 0
                    continue
    return arr


def boundary_v(mesh, arr):
    for node in mesh.nodes:
        if node.is_boundary():
            arr[node.id] = 0
    return arr


def boundary_pressure(mesh, arr):
    for node in mesh.nodes:
        if node.is_boundary():
            arr[node.id] = 0
    return arr


if __name__ == '__main__':
    mesh = Quadrangle2DMesh(60, 60, 60, 60)
    mesh.compute()
    timesteps = 100
    CFL = 0.8
    rho = 1
    mu = 0.01

    u = Variable2d(mesh, np.zeros(len(mesh.nodes)), boundary_u, "u")
    v = Variable2d(mesh, np.zeros(len(mesh.nodes)), boundary_v, "v")
    p = Variable2d(mesh, np.zeros(len(mesh.nodes)), boundary_pressure, "p")
    dt_var = DT(update_fn=DT.UpdatePolicies.CourantFriedrichsLewy, CFL=CFL, space_vars=[u, v])

    equation_system = [
        [d1t(u),        "=",    -u*d1dx(u) - v*d1dy(u) + laplass(u) * (mu/rho) ],
        [d1t(v),        "=",    -u*d1dx(v) - v*d1dy(v) + laplass(v) * (mu/rho) ],
        [laplass(p),    "=",    -rho / dt_var * (d1dx(u) + d1dy(v))                     ],
        [d1t(u),        "=",    -d1dx(p) / rho                                          ],
        [d1t(v),        "=",    -d1dy(p) / rho                                          ],
    ]

    equation = Equation(timesteps)
    history = equation.evaluate([u, v, p],
                                equation_system,
                                dt_var)


    fig, ax = plt.subplots()
    u_pic = np.zeros((mesh.x, mesh.y))
    v_pic = np.zeros((mesh.x, mesh.y))
    X, Y = np.meshgrid(np.linspace(0, mesh.x, mesh.x-1), np.linspace(0, mesh.y, mesh.y-1))

    def animate(i):
        print(i)

        data_u = history[0][i]
        data_v = history[1][i]
        for x_ in range(mesh.x):
            for y_ in range(mesh.y):
                u_pic[x_, y_] = data_u[mesh.coord_fo_idx(x_, y_)]
                v_pic[x_, y_] = data_v[mesh.coord_fo_idx(x_, y_)]


        u_curr_history = u_pic[1:-1, 1:-1]
        v_curr_history = v_pic[1:-1, 1:-1]
        ax.clear()
        ax.streamplot(X[1:, 1:], Y[1:, 1:], u_curr_history, v_curr_history)

    anim = animation.FuncAnimation(fig, animate, frames=timesteps, repeat=False)
    plt.show()