import numpy as np
import tiles3 as tiles

from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt


def f(X): return 3*np.sin(X) - 2*np.cos(X**2) - np.e * np.log(1 / (X + 3))
def _f(X, w): return w[X].sum()


def animate(i):
    ax.clear()
    ax.plot(x, y, label="$f(x)$", linestyle="--", color="r")
    ax.plot(x, _y[i, :], label="$\hat{f}(x)$", color="b")  # noqa: W605
    ax.set_title(f"Iteration {i}")
    ax.legend()
    ax.grid()


class Tile:

    iht_size = 2048

    def __init__(self, tiling_size, num_tilings):

        self.tiling_size = tiling_size
        self.num_tilings = num_tilings
        self.iht = tiles.IHT(self.iht_size)
        self.step = 0.1 / self.num_tilings

    def tile(self, x, range):

        for i, (min_x, max_x) in enumerate(range):
            x[i] = (x[i] - min_x) / (max_x - min_x)

        return tiles.tiles(self.iht, self.num_tilings, self.tiling_size*x)


if __name__ == "__main__":

    N_ITER = 501

    x = np.linspace(-2., 2., 100)
    y = f(x)
    _y = np.zeros(shape=(N_ITER, 100))

    tile = Tile(tiling_size=5, num_tilings=128)

    w = np.zeros(shape=2048)

    r = [(-2., 2.)]
    r_points = np.random.uniform(r[0][0], r[0][1], size=N_ITER)

    for k in range(N_ITER):
        r_tiles = tile.tile([r_points[k]], r)
        w[r_tiles] += tile.step * (f(r_points[k]) - _f(r_tiles, w))
        for i, value in enumerate(x):
            _y[k, i] = _f(tile.tile([value], r), w)

    fig, ax = plt.subplots(1, 1)

    ani = FuncAnimation(
        fig, animate, frames=N_ITER,
        interval=50, repeat=False
    )
    plt.show()
    plt.close()
