import numpy as np

from matplotlib import pyplot as plt


if __name__ == "__main__":

    N_EPOCHS = 1000

    init_rate, min_rate = 0.9, 0.01

    decay = 0.996

    ld = np.array(
        [max(init_rate-(1-decay)*i, min_rate) for i in range(N_EPOCHS)]
    )
    ed = np.array([max(init_rate*decay**i, min_rate) for i in range(N_EPOCHS)])

    plt.plot(ld, label="Linear decay", linestyle="-", color="b")
    plt.plot(ed, label="Exponential decay", linestyle="--", color="r")
    plt.title(f"Initial rate = {init_rate}")
    plt.legend()
    plt.grid()

    plt.show()
    plt.close()
