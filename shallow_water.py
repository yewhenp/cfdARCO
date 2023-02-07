import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from matplotlib import cm

from equation import Variable1d, Variable2d, Equation, d1dx, d2dx, d1dy, d2dy, dt, d2t



def boundary(arr):
    arr[0, :] = 0
    arr[:, -1] = 0
    arr[-1, :] = 0
    arr[:, 0] = 1
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

def boundary_pressure(arr):
    arr[0, :] = arr[1, :]
    arr[:, -1] = arr[:, -2]
    arr[-1, :] = arr[-2, :]
    arr[:, 0] = 0
    return arr


counter = 0

def boundary_h(arr):
    arr[0, :] = 0
    arr[-1, :] = 0
    arr[:, 0] = 0
    arr[:, -1] = 0

    global counter
    if counter < 5:
        counter += 1
        arr[9:11, 9:11] = 1

    return arr


if __name__ == '__main__':
    deltas = [1, 1]
    Lx = 20
    Ly = 20
    timesteps = 1000
    time_s = 10

    h = Variable2d(np.zeros((Lx, Ly)), boundary_h, deltas)
    u = Variable2d(np.zeros((Lx, Ly)), boundary_empty, deltas)
    v = Variable2d(np.zeros((Lx, Ly)), boundary_empty, deltas)

    g = 9.81  # Acceleration of gravity [m/s^2]
    H = 1  # Depth of fluid [m]
    f_0 = 1E-4  # Fixed part ofcoriolis parameter [1/s]
    beta = 2E-11  # gradient of coriolis parameter [1/ms]
    rho_0 = 1024.0  # Density of fluid [kg/m^3)]
    tau_0 = 0.1  # Amplitude of wind stress [kg/ms^2]
    k = 0.9
    y = np.linspace(-Ly / 2, Ly / 2, Ly)

    f = f_0 + beta * y  # Varying coriolis parameter
    L_R = np.sqrt(g * H) / f_0  # Rossby deformation radius
    c_R = beta * g * H / f_0 ** 2  # Long Rossby wave speed

    dt_ = time_s / timesteps

    alpha = dt_*f                # Parameter needed for coriolis scheme
    beta_c = alpha**2/4         # Parameter needed for coriolis scheme


    eq1 = (d1dx(h) * (-g) + u*beta_c*(-1) + v*alpha) * (1 / (1+beta_c))
    eq2 = (d1dx(h) * (-g) + v*beta_c*(-1) + u*alpha*(-1)) * (1 / (1+beta_c))
    eq3 = (d1dx(u) + d1dy(v)) * (-H)

    equation = Equation(timesteps = timesteps, time_s = time_s)
    history = equation.evaluate([u, v, h], [dt(u), dt(v), dt(h)], [eq1, eq2, eq3])

    X = np.arange(0, Lx, 1)
    Y = np.arange(0, Ly, 1)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = (ax.plot_surface(X, Y, history[2][0], cmap=cm.coolwarm, vmin=-1, vmax=1))

    ax.set_title('Wave')
    fig.colorbar(surf)  # Add a colorbar to the plot
    ax.set_zlim(-1.01, 1.01)


    def animate(i):
        print(i)
        ax.clear()
        surf = (ax.plot_surface(X, Y, history[2][i], cmap=cm.coolwarm, vmin=-1, vmax=1))
        ax.set_zlim(-1.01, 1.01)

        return surf

    anim = animation.FuncAnimation(fig, animate, frames=timesteps)
    plt.show()