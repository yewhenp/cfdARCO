import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from equation import Variable2d, Equation, d1dx, d2dx, d1dy, d2dy, d1t, DT, laplass


def boundary_u(arr):
    arr[:, 0] = 0
    arr[:, -1] = 0
    arr[-1, :] = 2 - arr[-2, :]
    arr[0, :] = - arr[1, :]
    return arr

def boundary_v(arr):
    arr[:, 0] = - arr[:, 1]
    arr[:, -1] = - arr[:, -2]
    arr[-1, :] = 0
    arr[0, :] = 0
    return arr

def boundary_pressure(arr):
    arr[:, 0] = 0
    arr[:, -1] = 0
    arr[-1, :] = 0
    arr[0, :] = 0
    return arr

def boundary_none(arr):
    return arr


if __name__ == '__main__':
    deltas = [1/10, 1/10]
    Lx = 11
    Ly = 11
    timesteps = 100
    CFL = 0.8
    rho = 1
    mu = 0.01

    u = Variable2d(np.zeros((Lx, Ly)), boundary_u, deltas, "u")
    v = Variable2d(np.zeros((Lx, Ly)), boundary_v, deltas, "v")
    p = Variable2d(np.zeros((Lx, Ly)), boundary_pressure, deltas, "p")
    dt_var = DT(update_fn=DT.UpdatePolicies.CourantFriedrichsLewy, CFL=CFL, space_vars=[u, v], deltas=deltas)

    equation_system = [
        [d1t(u),        "=",    -u*d1dx(u) - v*d1dy(u) + (d2dx(u) + d2dy(u)) * (mu/rho) ],
        [d1t(v),        "=",    -u*d1dx(v) - v*d1dy(v) + (d2dx(v) + d2dy(v)) * (mu/rho) ],
        [laplass(p),    "=",    -rho / dt_var * (d1dx(u) + d1dy(v))                     ],
        [d1t(u),        "=",    -d1dx(p) / rho                                          ],
        [d1t(v),        "=",    -d1dy(p) / rho                                          ],
    ]

    equation = Equation(timesteps)
    history = equation.evaluate([u, v, p],
                                equation_system,
                                dt_var)


    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.linspace(0, Lx, Lx-1), np.linspace(0, Ly, Ly-1))

    def animate(i):
        print(i)
        u_curr_history = history[0][i][1:-1, 1:-1]
        v_curr_history = history[1][i][1:-1, 1:-1]
        ax.clear()
        ax.streamplot(X[1:, 1:], Y[1:, 1:], u_curr_history, v_curr_history)

    anim = animation.FuncAnimation(fig, animate, frames=timesteps, repeat=False)
    plt.show()