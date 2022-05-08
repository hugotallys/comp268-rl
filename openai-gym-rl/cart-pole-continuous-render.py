import gym
import numpy as np
from tiles3 import tiles, IHT

DISCOUNT = 0.96  # gamma
EXPLORE_RATE = 0.15  # epsilon
LEARNING_RATE = 0.05  # alpha
EPISODES = (100, 100)
MAX_ITER = 500
RENDER = False

maxSize = 2048
iht = IHT(maxSize)
numTilings = 8
stepSize = 0.1/numTilings

weights = np.load("weights.npy")


def mytiles(X, tile_dim=10.0, min_x=-4., max_x=4.):
    scaleFactor = tile_dim / (max_x - min_x)
    return tiles(iht, numTilings, scaleFactor*X)


def v_hat(X):
    return weights[mytiles(X)].sum()


if __name__ == "__main__":

    cart_pole = gym.make("CartPole-v1")

    observation = cart_pole.reset()

    for t in range(MAX_ITER):
        cart_pole.render()
        curr_state = observation[2:]

        push_left = v_hat(np.concatenate([curr_state, [0]]))
        push_right = v_hat(np.concatenate([curr_state, [1]]))
        action = 1 if push_right > push_left else 0

        observation, reward, done, info = cart_pole.step(action)
        cart_pos = observation[0]
        if (
            cart_pos < cart_pole.observation_space.low[0]
            or cart_pos > cart_pole.observation_space.high[0]
        ):
            break

    cart_pole.close()
