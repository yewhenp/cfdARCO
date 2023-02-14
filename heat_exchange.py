import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from equation import Variable2d, Equation, d2dx, d2dy, d1t, DT


def boundary(arr):
    arr[10:20, 10:20] = -50
    arr[30:55, 30:55] = 100
    arr[0, :] = 0
    arr[-1, :] = 0
    arr[:, 0] = 0
    arr[:, -1] = 0
    return arr


if __name__ == '__main__':
    deltas = [1, 1]
    timesteps = 10000
    time_s = 500
    k = 5
    field = Variable2d(np.zeros((100, 100)), boundary, deltas)
    dt_var = DT(update_fn=DT.UpdatePolicies.constant_value, timesteps=timesteps, time_s=time_s)

    equation_system = [
        [d1t(field), "=", (d2dx(field) + d2dy(field)) * k],
    ]

    equation = Equation(timesteps = timesteps)
    history = equation.evaluate([field], equation_system, dt_var)[0]

    fig, ax = plt.subplots()
    sns.heatmap(history[0], vmax=100, vmin=-100, cmap="crest")

    def init():
        sns.heatmap(np.zeros_like(history[0]), vmax=100, vmin=-100, cbar=False, cmap=sns.color_palette("vlag", as_cmap=True))


    def animate(i):
        print(i)
        data = history[3*i]
        sns.heatmap(data, vmax=100, vmin=-100, square=True, cbar=False, cmap=sns.color_palette("vlag", as_cmap=True))


    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=1000, repeat=False, interval=20)
    plt.show()