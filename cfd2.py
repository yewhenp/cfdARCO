import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from equation import Variable1d, Variable2d, Equation, d1dx, d2dx, d1dy, d2dy, dt, d2t



def boundary_u(arr):
    arr[:, 0] = 0
    arr[:, -1] = 0
    # arr[-1, :] = 2 - arr[-2, :]
    # arr[0, :] = arr[1, :]
    arr[-1, :] = 0
    arr[0, :] = 1
    return arr

def boundary_v(arr):
    # arr[:, 0] = -arr[:, 1]
    # arr[:, -1] = arr[:, -2]
    arr[:, 0] = 0
    arr[:, -1] = 0
    arr[-1, :] = 0
    arr[0, :] = 0
    return arr

def boundary_pressure(arr):
    arr[0, :] = arr[1, :]
    arr[:, -1] = arr[:, -2]
    arr[-1, :] = 0
    arr[:, 0] = arr[:, 1]
    return arr

def boundary2(arr):
    arr[-1, :] = 0
    arr[0, :] = 0
    arr[:, 0] = 0
    return arr

def boundary_ones(arr):
    arr[0, :] = 1
    arr[-1, :] = 1
    arr[:, 0] = 1
    arr[:, -1] = 1
    return arr

def boundary_empty(arr):
    arr[0, :] = 0
    arr[-1, :] = 0
    arr[:, 0] = 0
    arr[:, -1] = 0
    return arr


def boundary_none(arr):
    return arr


if __name__ == '__main__':
    deltas = [0.1, 0.1]
    Lx = 10
    Ly = 10
    timesteps = 100
    time_s = 150

    rho = 1
    mu = 0.01

    u = Variable2d(np.zeros((Lx, Ly)), boundary_u, deltas)
    v = Variable2d(np.zeros((Lx, Ly)), boundary_v, deltas)
    p = Variable2d(np.zeros((Lx, Ly)), boundary_empty, deltas)


    eq1 = u*d1dx(u)*(-1) + v*d1dy(u)*(-1) + (d2dx(u) + d2dy(u)) * (mu/rho)
    eq2 = u*d1dx(v)*(-1) + v*d1dy(v)*(-1) + (d2dx(v) + d2dy(v)) * (mu/rho)
    eq3 = (d1dx(u) + d1dy(v)) * (-rho/(time_s/timesteps))
    eq4 = d1dx(p) * (-1/rho)
    eq5 = d1dy(p) * (-1/rho)

    equation = Equation(timesteps = timesteps, time_s = time_s)
    history = equation.evaluate([u, v, p, u, v],
                                [dt(u), dt(v), d2dx(p)+d2dy(p), dt(u), dt(v)],
                                [eq1, eq2, eq3, eq4, eq5])

    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.linspace(0, Lx, Lx+1), np.linspace(0, Ly, Ly+1))

    def animate(i):
        print(i)
        u_curr_history = history[0][i]
        v_curr_history = history[1][i]
        # print(u_curr_history.mean())
        # print(v_curr_history.mean())
        ax.clear()
        ax.quiver(X[1:, 1:], Y[1:, 1:], u_curr_history, v_curr_history)

    anim = animation.FuncAnimation(fig, animate, frames=timesteps, repeat=False)
    plt.show()