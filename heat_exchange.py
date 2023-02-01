import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from equation import Variable1d, Variable2d, Equation, d1dx, d2dx, d1dy, d2dy, dt



def boundary_1(arr):
    arr[0, :] = 0
    arr[-1, :] = -100
    arr[:, 0] = -50
    arr[:, -1] = 0
    return arr


def boundary_2(arr):
    arr[10:20, 10:20] = -50
    arr[30:55, 30:55] = 100
    arr[0, :] = 0
    arr[-1, :] = 0
    arr[:, 0] = 0
    arr[:, -1] = 0
    return arr


if __name__ == '__main__':
    initial = np.zeros((100, 100))
    boundary_conditions = boundary_2
    deltas = [1, 1]
    field = Variable2d(initial, boundary_conditions, deltas)

    k = 5
    eq_formula = (d2dx(field) + d2dy(field)) * k

    equation = Equation(timesteps = 10000, time_s = 500)
    history = equation.evaluate([field], [dt(field)], [eq_formula])[0]

    fig, ax = plt.subplots()
    sns.heatmap(history[0], vmax=100, vmin=-100, cmap="crest")

    def init():
        sns.heatmap(np.zeros_like(history[0]), vmax=100, vmin=-100, cbar=False, cmap=sns.color_palette("vlag", as_cmap=True))


    def animate(i):
        print(i)
        data = history[10 * i]
        sns.heatmap(data, vmax=100, vmin=-100, square=True, cbar=False, cmap=sns.color_palette("vlag", as_cmap=True))


    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=1000, repeat=False, interval=20)
    plt.show()