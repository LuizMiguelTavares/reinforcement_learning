# draw_obs_grid.py

import numpy as np
import matplotlib.pyplot as plt


def draw_obs_grid(nx, ny):
    obstacle_map = np.zeros((ny, nx), dtype=int)
    fig, ax = plt.subplots(figsize=(8, 8))

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return

        ix, iy = int(round(event.xdata)), int(round(event.ydata))

        if 0 <= ix < nx and 0 <= iy < ny:
            obstacle_map[iy, ix] = 1 - obstacle_map[iy, ix]

            ax.clear()
            ax.imshow(obstacle_map, cmap='Reds', vmin=0, vmax=1)

            ax.set_xticks(np.arange(-.5, nx, 1), minor=True)
            ax.set_yticks(np.arange(-.5, ny, 1), minor=True)
            ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
            ax.tick_params(which="minor", size=0)
            ax.set_xticks(np.arange(0, nx, 1))
            ax.set_yticks(np.arange(0, ny, 1))
            ax.set_title(
                "Defina os obstáculos. Feche a janela para continuar.")
            fig.canvas.draw()

    ax.imshow(obstacle_map, cmap='Reds', vmin=0, vmax=1)
    ax.set_xticks(np.arange(-.5, nx, 1), minor=True)
    ax.set_yticks(np.arange(-.5, ny, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks(np.arange(0, nx, 1))
    ax.set_yticks(np.arange(0, ny, 1))
    ax.set_title("Defina os obstáculos. Feche a janela para continuar.")

    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    return obstacle_map
