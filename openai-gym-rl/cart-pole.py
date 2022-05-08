import gym
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange

DISCOUNT = 0.99  # gamma
EXPLORE_RATE = 0.3  # epsilon
LEARNING_RATE = 0.1  # alpha
EPISODES = (500, 50)
MAX_ITER = 250
RENDER = False


def discretize_state(obs, n_bins):
    obs_bins = [
        np.linspace(low, high, n_bins[i])
        for i, (low, high) in enumerate(
            zip(
                low_cart,
                high_cart
            )
        )
    ]
    return np.array([np.digitize(o, obs_bins[i]) for i, o in enumerate(obs)])


if __name__ == "__main__":

    cart_pole = gym.make("CartPole-v1")

    low_cart = cart_pole.observation_space.low
    low_cart[2] = 0.25 * low_cart[2]
    low_cart[3] = -4.

    high_cart = cart_pole.observation_space.high
    high_cart[2] = 0.25 * high_cart[2]
    high_cart[3] = 4.

    n_bins = (0, 0, 40, 40)
    Q = np.zeros(
        shape=(42, 42) + (cart_pole.action_space.n,)
    )

    accumulated_return = np.zeros(shape=EPISODES[0])

    episodes_loop = trange(EPISODES[0])

    for i_episode in episodes_loop:
        acc_ret = np.zeros(shape=EPISODES[1])
        for i in range(EPISODES[1]):
            observation = cart_pole.reset()
            for t in range(MAX_ITER):
                cart_pole.render() if RENDER else None
                curr_state = discretize_state(observation, n_bins)
                _, _, cs1, cs2 = curr_state
                if np.random.uniform() < EXPLORE_RATE:
                    action = cart_pole.action_space.sample()
                else:
                    action = np.argmax(Q[cs1, cs2, :])
                observation, reward, done, info = cart_pole.step(action)
                next_state = discretize_state(observation, n_bins)
                _, _, ns1, ns2 = next_state

                if done:
                    reward *= -2.

                Q[cs1, cs2, action] += LEARNING_RATE * (
                    reward + DISCOUNT * np.max(
                        Q[ns1, ns2, :]
                    ) - Q[cs1, cs2, action]
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

    np.save('qtable', Q)

    plt.figure()
    plt.subplot(1, 2, 1)
    sns.heatmap(Q[:, :, 0])
    plt.ylabel("$\\theta$")
    plt.xlabel("$\\frac{d\\theta}{dt}$")
    plt.title("Q(s, 0)")
    plt.subplot(1, 2, 2)
    sns.heatmap(Q[:, :, 1])
    plt.xlabel("$\\frac{d\\theta}{dt}$")
    plt.title("Q(s, 1)")
    plt.savefig("state-action.png")
    plt.figure(figsize=(16, 8))
    plt.plot(accumulated_return)
    plt.xlabel("epoch")
    plt.title("Mean accumulated return")
    plt.savefig("acc-return.png")
    plt.show()
