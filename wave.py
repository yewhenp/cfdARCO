import math
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pygame
from matplotlib import cm
import seaborn as sns
from equation import Variable1d, Variable2d, Equation, d1dx, d2dx, d1dy, d2dy, dt, d2t, laplass


counter = 0

def place_raindrops(arr):
    if random.random()<0.02:
        sz = 2
        sigma = 1.4
        xx, yy = np.meshgrid(range(-sz, sz), range(-sz, sz))
        gauss_peak = 300 / (sigma * 2 * math.pi) * (math.sqrt(2 * math.pi)) * np.exp(
            - 0.5 * ((xx ** 2 + yy ** 2) / (sigma ** 2)))

        w, h = gauss_peak.shape
        x = random.randrange(w, arr.shape[0]-w)
        y = random.randrange(h, arr.shape[1]-h)

        height = 0.01
        arr[x:x+w, y:y+h] += gauss_peak * height
        # arr[x-w-dist:x+w-dist, y-h:y+h] += height * gauss_peak
        # arr[x-w:x+w, y-h+dist:y+h+dist] -= height * gauss_peak
        # arr[x-w+dist:x+w+dist, y-h:y+h] += height * gauss_peak
        # arr[x-w:x+w, y-h-dist:y+h-dist] -= height * gauss_peak

ii = 0

def boundary(arr):
    global ii

    arr[0, :] = 0
    arr[:, 0] = 0
    arr[:, -1] = 0
    arr[-1, :] = 0
    # arr[0:5, :] = np.sin(ii * 0.15) * 20

    ii+=1
    place_raindrops(arr)

    # global counter
    # if counter < 10:
    #     if counter > 5:
    #         print("here")
    #         arr[30, 30] = 1
    #     counter += 1

    return arr


# def bound_2(arr):
#     c = dimx - 1
#     sz = 1
#     arr_n = np.zeros_like(arr)
#     arr_n[dimx - sz - 1:c, 1:dimy - 1] = arr[dimx - sz - 2:c - 1, 1:dimy - 1] + (
#                 kappa[dimx - sz - 1:c, 1:dimy - 1] - 1) / (kappa[dimx - sz - 1:c, 1:dimy - 1] + 1) * (
#                                                     arr[dimx - sz - 2:c - 1, 1:dimy - 1] - arr[dimx - sz - 1:c,
#                                                                                             1:dimy - 1])
#
#     c = 0
#     arr_n[c:sz, 1:dimy - 1] = arr[c + 1:sz + 1, 1:dimy - 1] + (kappa[c:sz, 1:dimy - 1] - 1) / (
#                 kappa[c:sz, 1:dimy - 1] + 1) * (arr[c + 1:sz + 1, 1:dimy - 1] - arr[c:sz, 1:dimy - 1])
#
#     r = dimy - 1
#     arr_n[1:dimx - 1, dimy - 1 - sz:r] = arr[1:dimx - 1, dimy - 2 - sz:r - 1] + (
#                 kappa[1:dimx - 1, dimy - 1 - sz:r] - 1) / (kappa[1:dimx - 1, dimy - 1 - sz:r] + 1) * (
#                                                     arr[1:dimx - 1, dimy - 2 - sz:r - 1] - arr[1:dimx - 1,
#                                                                                             dimy - 1 - sz:r])
#
#     r = 0
#     arr_n[1:dimx - 1, r:sz] = arr[1:dimx - 1, r + 1:sz + 1] + (kappa[1:dimx - 1, r:sz] - 1) / (
#                 kappa[1:dimx - 1, r:sz] + 1) * (arr[1:dimx - 1, r + 1:sz + 1] - arr[1:dimx - 1, r:sz])
#
#     return arr_n


def I(x, y):
    """Gaussian peak at (Lx/2, Ly/2)."""
    return np.exp(-0.5*(x-Lx/2.0)**2 - 0.5*(y-Ly/2.0)**2)


if __name__ == '__main__':

    Lx = 300
    Ly = 300

    dimx = Lx
    dimy = Ly

    scale = 2
    timesteps = 1000
    time_s = 1000

    ts = time_s / timesteps

    initial = np.zeros((Lx, Ly))
    velocity = np.zeros((Lx, Ly))

    velocity[:,:] = 0.3            # 0.39 m/s Wave velocity of shallow water waves (lambda 0.1, depth 0.1)
    # velocity[220:300,100:dimy-100] = 0.2      # will be set to a constant value of tau
    # velocity[300:400,100:dimy-100] = 0.1      # will be set to a constant value of tau
    # velocity[0:dimx, 300:] = 0.4     # will be set to a constant value of tau

    # compute tau and kappa from the velocity field
    tau = ( velocity*ts )**2
    kappa = 0.3

    deltas = [1, 1]
    field = Variable2d(initial, boundary, deltas)

    eq = (d2dx(field) + d2dy(field)) * tau
    # tau[0, :] = 1
    # tau[-1, :] = 1
    # tau[:, 0] = 1
    # tau[:, -1] = 1
    # eq = laplass(field) * tau

    equation = Equation(timesteps = timesteps, time_s = time_s)
    history = equation.evaluate([field], [d2t(field)], [eq])[0]


    pygame.init()
    display = pygame.display.set_mode((Lx*scale, Ly*scale))
    pygame.display.set_caption("Solving the 2d Wave Equation")
    pixeldata = np.zeros((Lx, Ly, 3), dtype=np.uint8)

    i = 2

    while True:
        print(i)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                break

        dimx = Lx
        dimy = Ly
        u = np.asarray([history[i+1], history[i], history[i-1]])
        # pixeldata = np.zeros((Lx, Ly, 3), dtype=np.uint8)
        pixeldata[1:dimx, 1:dimy, 0] = np.clip((u[0, 1:dimx, 1:dimy]>0) * 20 * u[0, 1:dimx, 1:dimy]+u[1, 1:dimx, 1:dimy]+u[2, 1:dimx, 1:dimy], 0, 255)
        pixeldata[1:dimx, 1:dimy, 1] = np.clip((u[0, 1:dimx, 1:dimy]>0) * 20 * u[0, 1:dimx, 1:dimy]+u[1, 1:dimx, 1:dimy]+u[2, 1:dimx, 1:dimy], 0, 255)
        pixeldata[1:dimx, 1:dimy, 2] = np.clip((u[0, 1:dimx, 1:dimy]>0) * 20 * u[0, 1:dimx, 1:dimy]+u[1, 1:dimx, 1:dimy]+u[2, 1:dimx, 1:dimy], 0, 255)

        surf = pygame.surfarray.make_surface(pixeldata)
        display.blit(pygame.transform.scale(surf, (Lx * scale, Ly * scale)), (0, 0))
        pygame.display.update()

        i+=1
        # time.sleep(0.2)


    # X = np.arange(0, Lx, 1)
    # Y = np.arange(0, Ly, 1)
    # X, Y = np.meshgrid(X, Y)
    #
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    # surf = (ax.plot_surface(X, Y, history[0], cmap=cm.coolwarm, vmin=-1, vmax=1))
    #
    # ax.set_title('Wave')
    # fig.colorbar(surf)  # Add a colorbar to the plot
    # ax.set_zlim(-1.01, 1.01)
    #
    #
    # def animate(i):
    #     print(i)
    #     ax.clear()
    #     surf = (ax.plot_surface(X, Y, history[i], cmap=cm.coolwarm, vmin=-1, vmax=1))
    #     ax.set_zlim(-1.01, 1.01)
    #
    #     return surf
    #
    # anim = animation.FuncAnimation(fig, animate, frames=timesteps)
    # plt.show()