import gym
import sys
import numpy as np

MAX_ITER = 500


def discretize_state(obs, n_bins):
    obs_bins = [
        np.linspace(low, high, n_bins[i])
        for i, (low, high) in enumerate(zip(low_cart, high_cart))
    ]
    return np.array([np.digitize(o, obs_bins[i]) for i, o in enumerate(obs)])


if __name__ == "__main__":

    cart_pole = gym.make("CartPole-v1")

    low_cart = cart_pole.observation_space.low
    low_cart[2] = 0.25 * low_cart[2]
    low_cart[3] = -4.0

    high_cart = cart_pole.observation_space.high
    high_cart[2] = 0.25 * high_cart[2]
    high_cart[3] = 4.0

    n_bins = (0, 0, 40, 40)
    Q = np.load("qtable.npy")

    observation = cart_pole.reset()

    mode = None
    try:
        mode = sys.argv[1]
    except IndexError:
        mode = "t"

    for _ in range(MAX_ITER):
        cart_pole.render()
        curr_state = discretize_state(observation, n_bins)
        _, _, cs1, cs2 = curr_state
        if mode == "r":
            action = cart_pole.action_space.sample()
        else:
            action = np.argmax(Q[cs1, cs2, :])
        observation, reward, done, info = cart_pole.step(action)
        next_state = discretize_state(observation, n_bins)
        _, _, ns1, ns2 = next_state
        cart_pos = observation[0]
        if (
            cart_pos < cart_pole.observation_space.low[0]
            or cart_pos > cart_pole.observation_space.high[0]
        ):
            break

    cart_pole.close()
