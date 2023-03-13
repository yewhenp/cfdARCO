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
                    arr[node.id] = 0
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

def boundary_none(mesh, arr):
    return arr



def initial_val_aaa(mesh, val_out, val_in):
    arr = np.zeros(len(mesh.nodes))
    for node in mesh.nodes:
        if .30 < node.y < .70:
            arr[node.id] = val_in
        else:
            arr[node.id] = val_out
    return arr


def initial_pertrbations(mesh):
    arr = np.zeros(len(mesh.nodes))
    for node in mesh.nodes:
        if .30 < node.x < .70:
            arr[node.id] = -0.3
        else:
            arr[node.id] = -0.5
    return arr


def boundary_aaa(mesh, arr, val_in, val_out):
    for node in mesh.nodes:
        if node.is_boundary():
            if .30 < node.y < .70:
                arr[node.id] = val_in
            else:
                arr[node.id] = val_out
    return arr


def boundary_pertrbations(mesh, arr):
    for node in mesh.nodes:
        if node.is_boundary():
            if .30 < node.x < .70:
                arr[node.id] = -0.3
            else:
                arr[node.id] = -0.5
    return arr


if __name__ == '__main__':
    mesh = Quadrangle2DMesh(50, 50, 50, 50)
    mesh.compute()

    timesteps = 15
    CFL = 0.5
    gamma = 5/3

    rho = Variable2d(mesh, initial_val_aaa(mesh, 1, 2), lambda mesh, arr: boundary_aaa(mesh, arr, 1, 2), "rho")
    u = Variable2d(mesh, initial_val_aaa(mesh, -0.5, 0.5), lambda mesh, arr: boundary_aaa(mesh, arr, -0.5, 0.5), "u")
    v = Variable2d(mesh, initial_pertrbations(mesh), boundary_pertrbations, "v")
    p = Variable2d(mesh, 2.5 * np.ones(len(mesh.nodes)), lambda mesh, arr: boundary_aaa(mesh, arr, 2.5, 2.5), "p")

    E_c = p.current / (gamma - 1) + 0.5 * rho.current * (u.current ** 2 + v.current ** 2)

    mass = Variable2d(mesh, initial_val_aaa(mesh, 1, 2), boundary_none, "mass")
    rho_u = Variable2d(mesh, rho.current*u.current, boundary_none, "rho_u")
    rho_v = Variable2d(mesh, rho.current*u.current, boundary_none, "rho_v")
    rho_e = Variable2d(mesh, E_c, boundary_none, "rho_e")
    indicator = Variable2d(mesh, np.zeros(len(mesh.nodes)), boundary_none, "indicator")
    dt = DT(update_fn=DT.UpdatePolicies.CourantFriedrichsLewy, CFL=CFL, space_vars=[u, v, p, rho, gamma])

    E = p / (gamma - 1) + 0.5 * rho * (u*u + v*v)
    E_ = p / (gamma - 1) + 0.5 * mass * (rho_u*rho_u/(rho*rho) + rho_v*rho_v/(rho*rho))


    equation_system = [
        # [rho, "=", rho - 0.5 * dt * (u*d1dx(rho) + rho*d1dx(u) + v*d1dy(rho) + rho*d1dy(v))],
        # [u, "=", u - 0.5 * dt * (u*d1dx(u) + v*d1dy(v) + (1/rho)*d1dx(p))],
        # [v, "=", v - 0.5 * dt * (u*d1dx(v) + v*d1dy(v) + (1/rho)*d1dy(p))],
        # [p, "=", p - 0.5 * dt * (gamma * p * (d1dx(u) + d1dy(v)) + u*d1dx(p) + v*d1dy(p))],

        # [indicator, "=", rho * mesh.volumes],
        # [d1t(indicator), "=", - d1dy(rho*v)],

        [mass, "=", rho * mesh.volumes],
        [rho_u, "=", rho * u * mesh.volumes],
        [rho_v, "=", rho * v * mesh.volumes],
        [rho_e, "=", E * mesh.volumes],

        [d1t(mass), "=", (d1dx(rho_u) + d1dy(rho_v))],
        [d1t(rho_u),    "=", (d1dx(rho_u*rho_u/mass + p)  + d1dy(rho_v*rho_u/mass)) ],
        [d1t(rho_v),    "=", (d1dx(rho_v*rho_u/mass)  + d1dy(rho_v*rho_v/mass + p)) ],
        [d1t(rho_e), "=", (d1dx((E_ + p) * (rho_u / rho)) + d1dy((E_ + p) * (rho_v / rho))) ],

        [rho, "=", mass / mesh.volumes],
        [u, "=", rho_u / u / mesh.volumes ],
        [v, "=", rho_v / v / mesh.volumes ],
        [p, "=", (rho_e / mesh.volumes - 0.5 * rho * (u * u + v * v)) * (gamma - 1)],
    ]


    equation = Equation(timesteps)
    equation.evaluate([rho, u, v, p], equation_system, dt)


    fig, ax = plt.subplots()

    def animate(i):
        print(i)
        rho_pic = np.zeros((mesh.x, mesh.y))
        data_rho = rho.history[i]
        for x_ in range(mesh.x):
            for y_ in range(mesh.y):
                rho_pic[x_, y_] = data_rho[mesh.coord_fo_idx(x_, y_)]

        ax.clear()
        ax.imshow(rho_pic)

    anim = animation.FuncAnimation(fig, animate, frames=timesteps, repeat=False)
    anim.save("cfd.gif")
    plt.show()
