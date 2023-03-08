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
                if vertex.x < 1.5:
                    arr[node.id] = 0
                    continue
                else:
                    arr[node.id] = 0
                    continue
    return arr


def boundary_v(mesh, arr):
    for node in mesh.nodes:
        if node.is_boundary():
            for vertex_id in node.vertexes_id:
                vertex = mesh.vertexes[vertex_id]
                if vertex.x < 1.5:
                    arr[node.id] = 1
                    continue
                else:
                    arr[node.id] = 0
                    continue
    return arr


def boundary_pressure(mesh, arr):
    for node in mesh.nodes:
        if node.is_boundary():
            arr[node.id] = 0
    return arr

def boundary_rho(mesh, arr):
    for node in mesh.nodes:
        if node.is_boundary():
            arr[node.id] = 1
    return arr

if __name__ == '__main__':
    mesh = Quadrangle2DMesh(10, 10, 10, 10)
    mesh.compute()

    timesteps = 100
    CFL = 0.8
    gamma = 5/3

    rho = Variable2d(mesh, np.ones(len(mesh.nodes)), boundary_rho, "rho")
    u = Variable2d(mesh, np.zeros(len(mesh.nodes)), boundary_u, "u")
    v = Variable2d(mesh, np.zeros(len(mesh.nodes)), boundary_v, "v")
    p = Variable2d(mesh, np.zeros(len(mesh.nodes)), boundary_pressure, "p")
    dt_var = DT(update_fn=DT.UpdatePolicies.CourantFriedrichsLewy, CFL=CFL, space_vars=[u, v])

    equation_system = [
        [d1t(rho), "=", - u * d1dx(rho) - rho * d1dx(u) - v * d1dy(rho) - rho * d1dy(v)],
        [d1t(u),        "=",    - u*d1dx(u) - (1/rho)*d1dx(p)  - v*d1dy(u)                      ],
        [d1t(v),        "=",    - u*d1dx(v) - v*d1dy(v) - (1/rho)*d1dy(p)                       ],
        [d1t(p), "=", - gamma * p * d1dx(u) - u * d1dx(p) - gamma * p * d1dy(v) - v * d1dx(p)],
    ]

    equation = Equation(timesteps)
    equation.evaluate([rho, u, v, p], equation_system, dt_var)


    fig, ax = plt.subplots()
    u_pic = np.zeros((mesh.x, mesh.y))
    v_pic = np.zeros((mesh.x, mesh.y))
    rho_pic = np.zeros((mesh.x, mesh.y))

    def animate(i):
        print(i)

        data_u = u.history[i]
        data_v = v.history[i]
        data_rho = rho.history[i]
        for x_ in range(mesh.x):
            for y_ in range(mesh.y):
                u_pic[x_, y_] = data_u[mesh.coord_fo_idx(x_, y_)]
                v_pic[x_, y_] = data_v[mesh.coord_fo_idx(x_, y_)]
                rho_pic[x_, y_] = data_rho[mesh.coord_fo_idx(x_, y_)]

        u_curr_history = u_pic[1:-1, 1:-1]
        v_curr_history = v_pic[1:-1, 1:-1]
        ax.clear()
        ax.imshow(rho_pic)

    anim = animation.FuncAnimation(fig, animate, frames=timesteps, repeat=False)
    plt.show()