import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fvm import Variable2d, Equation, d1dx, d1dy, d1t, DT, stab_x, stab_y, to_grid
from mesh import Quadrangle2DMesh


def boundary_none(mesh, arr):
    return arr



def initial_val_aaa(mesh, val_out, val_in):
    arr = np.zeros(len(mesh.nodes), dtype=np.float16)
    for node in mesh.nodes:
        if .30 < node.y < .70:
            arr[node.id] = val_in
        else:
            arr[node.id] = val_out
    return arr


def initial_pertrbations(mesh):
    arr = np.zeros(len(mesh.nodes), dtype=np.float16)
    for node in mesh.nodes:
        if .30 < node.x < .70:
            arr[node.id] = -0.3
        else:
            arr[node.id] = -0.5
    return arr


def boundary_aaa(mesh, arr, val_out, val_in):
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


def boundary_copy(mesh, arr, copy_var):
    for node in mesh.nodes:
        if node.is_boundary():
            arr[node.id] = copy_var[node.id]
    return arr


if __name__ == '__main__':
    L = 50

    mesh = Quadrangle2DMesh(L, L, 1, 1)
    mesh.compute()

    timesteps = 30
    CFL = 0.5
    gamma = 5/3

    rho_initial = initial_val_aaa(mesh, 1, 2)
    rho = Variable2d(mesh, rho_initial, lambda mesh, arr: boundary_copy(mesh, arr, rho_initial.copy()), "rho")

    u_initial = initial_val_aaa(mesh, -0.5, 0.5)
    u = Variable2d(mesh, u_initial, lambda mesh, arr: boundary_copy(mesh, arr, u_initial.copy()), "u")

    v_initial = initial_pertrbations(mesh)
    v = Variable2d(mesh, v_initial, lambda mesh, arr: boundary_copy(mesh, arr, v_initial.copy()), "v")

    p_initial = 2.5 * np.ones(len(mesh.nodes))
    p = Variable2d(mesh, p_initial, lambda mesh, arr: boundary_copy(mesh, arr, p_initial.copy()), "p")

    rho_t_h = Variable2d(mesh, np.zeros_like(rho_initial, dtype=np.float16), boundary_none, "rho")
    u_t_h = Variable2d(mesh, np.zeros_like(rho_initial, dtype=np.float16), boundary_none, "u")
    v_t_h = Variable2d(mesh, np.zeros_like(rho_initial, dtype=np.float16), boundary_none, "v")
    p_t_h = Variable2d(mesh, np.zeros_like(rho_initial, dtype=np.float16), boundary_none, "p")

    mass_initial = rho.current * mesh.volumes
    mass = Variable2d(mesh, mass_initial, lambda mesh, arr: boundary_copy(mesh, arr, mass_initial.copy()), "mass")

    rho_u_initial = rho.current * u.current * mesh.volumes
    rho_u = Variable2d(mesh, rho_u_initial, lambda mesh, arr: boundary_copy(mesh, arr, rho_u_initial.copy()), "rho_u")

    rho_v_initial = rho.current * v.current * mesh.volumes
    rho_v = Variable2d(mesh, rho_v_initial, lambda mesh, arr: boundary_copy(mesh, arr, rho_v_initial.copy()), "rho_v")

    E = p / (gamma - 1) + 0.5 * rho * (u * u + v * v)
    E_initial = (p.current / (gamma - 1) + 0.5 * rho.current * (u.current * u.current + v.current * v.current)) * mesh.volumes
    rho_e = Variable2d(mesh, E_initial, lambda mesh, arr: boundary_copy(mesh, arr, E_initial.copy()), "rho_e")

    dt = DT(update_fn=DT.UpdatePolicies.CourantFriedrichsLewy, CFL=CFL, space_vars=[u, v, p, rho, gamma, L])

    equation_system = [
        [rho, "=", mass / mesh.volumes],
        [u, "=", rho_u / rho / mesh.volumes],
        [v, "=", rho_v / rho / mesh.volumes],
        [p, "=", (rho_e / mesh.volumes - 0.5 * rho * (u * u + v * v)) * (gamma - 1)],

        [rho_t_h, "=", rho - 0.5 * dt * (u * rho.dx + rho * u.dx + v * rho.dy + rho * v.dy)],
        [u_t_h, "=", u - 0.5 * dt * (u * u.dx + v * u.dy + (1/rho) * p.dx)],
        [v_t_h, "=", v - 0.5 * dt * (u * v.dx + v * v.dy + (1/rho) * p.dy)],
        [p_t_h, "=", p - 0.5 * dt * (gamma * p * (u.dx + v.dy) + u * p.dx + v * p.dy)],

        [rho, "=", rho_t_h],
        [u, "=", u_t_h],
        [v, "=", v_t_h],
        [p, "=", p_t_h],

        [d1t(mass), "=", -((d1dx(rho*u) + d1dy(rho*v)) - (stab_x(rho) + stab_y(rho)))],
        [d1t(rho_u),    "=", -((d1dx(rho*u*u + p)  + d1dy(rho*v*u)) - (stab_x(rho*u) + stab_y(rho*u))) ],
        [d1t(rho_v),    "=", -((d1dx(rho*v*u)  + d1dy(rho*v*v + p)) - (stab_x(rho*v) + stab_y(rho*v))) ],
        [d1t(rho_e), "=", -((d1dx((E + p) * u) + d1dy((E + p) * v)) - (stab_x(E) + stab_y(E))) ],

        [rho, "=", mass / mesh.volumes],
        [u, "=", rho_u / rho / mesh.volumes],
        [v, "=", rho_v / rho / mesh.volumes],
        [p, "=", (rho_e / mesh.volumes - 0.5 * rho * (u * u + v * v)) * (gamma - 1)],
    ]


    equation = Equation(timesteps)
    equation.evaluate([rho, u, v, p, mass, rho_u, rho_v, rho_e], equation_system, dt)


    fig, ax = plt.subplots()

    def animate(i):
        print(i)
        rho_pic = np.zeros((mesh.x, mesh.y), dtype=np.float64)
        data_rho = rho.history[i]
        for x_ in range(mesh.x):
            for y_ in range(mesh.y):
                rho_pic[x_, y_] = data_rho[mesh.coord_fo_idx(x_, y_)]

        ax.clear()
        ax.imshow(rho_pic)

    anim = animation.FuncAnimation(fig, animate, frames=timesteps, repeat=False)
    anim.save("cfd.gif")
    plt.show()
