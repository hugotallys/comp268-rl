import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from tiles3 import tiles, IHT

DISCOUNT = 0.96  # gamma
EXPLORE_RATE = 0.15  # epsilon
LEARNING_RATE = 0.05  # alpha
EPISODES = (100, 100)
MAX_ITER = 250
RENDER = False

maxSize = 2048
iht = IHT(maxSize)
numTilings = 8
stepSize = 0.1/numTilings

weights = np.zeros(shape=maxSize)


def mytiles(X, tile_dim=10.0, min_x=-4., max_x=4.):
    scaleFactor = tile_dim / (max_x - min_x)
    return tiles(iht, numTilings, scaleFactor*X)


def v_hat(X):
    return weights[mytiles(X)].sum()


if __name__ == "__main__":

    cart_pole = gym.make("CartPole-v1")

    low_cart = cart_pole.observation_space.low
    low_cart[2] = 0.25 * low_cart[2]
    low_cart[3] = -4.

    high_cart = cart_pole.observation_space.high
    high_cart[2] = 0.25 * high_cart[2]
    high_cart[3] = 4.

    accumulated_return = np.zeros(shape=EPISODES[0])

    episodes_loop = trange(EPISODES[0])

    for i_episode in episodes_loop:
        acc_ret = np.zeros(shape=EPISODES[1])
        for i in range(EPISODES[1]):
            observation = cart_pole.reset()
            for t in range(MAX_ITER):
                cart_pole.render() if RENDER else None
                curr_state = observation[2:]
                if np.random.uniform() < EXPLORE_RATE:
                    action = cart_pole.action_space.sample()
                else:
                    push_left = v_hat(np.concatenate([curr_state, [0]]))
                    push_right = v_hat(np.concatenate([curr_state, [1]]))
                    action = 1 if push_right > push_left else 0
                observation, reward, done, info = cart_pole.step(action)
                next_state = observation[2:]

                if done:
                    reward *= -2.

                state_tiles = mytiles(np.concatenate([curr_state, [action]]))
                v_hat_next = np.max(
                    np.array(
                        [
                            v_hat(np.concatenate(
                                [next_state, [a]])) for a in range(2)
                        ]
                    )
                )

                weights[state_tiles] += stepSize * (
                    reward + DISCOUNT * v_hat_next - v_hat(
                        np.concatenate([curr_state, [action]])
                    )
                )

                if done:
                    acc_ret[i] = t + 1
                    episodes_loop.set_description(
                        desc="Episode {0:0>4} fineshed after {1:0>3} \
                            timesteps".format(
                            i_episode + 1, t + 1
                        )
                    )
                    break
        accumulated_return[i_episode] = acc_ret.mean()

    cart_pole.close() if RENDER else None

    np.save('weights', weights)

    plt.figure(figsize=(16, 8))
    plt.plot(accumulated_return)
    plt.xlabel("epoch")
    plt.title("Mean accumulated return")
    plt.savefig("acc-return-cont.png")
    plt.show()
