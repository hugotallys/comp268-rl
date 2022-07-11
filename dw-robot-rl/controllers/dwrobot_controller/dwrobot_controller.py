import numpy as np

from controller import Robot  # type: ignore
from tiles3 import tiles, IHT

DISCOUNT = 0.96
EXPLORE_RATE = 0.18

maxSize = 2**20
iht = IHT(maxSize)
numTilings = 64
stepSize = 0.1/numTilings

weights = np.zeros(shape=maxSize)


def mytiles(s, a, tile_dim=20):
    s = s.copy()

    a /= 2

    s[-5:-3] = (s[-5:-3] + 0.05) / (2 * 0.05)
    s[-3] = (0.6 + s[-3]) / (2 * 0.6)
    s[-2] *= 2
    s[-1] = (s[-1] + np.pi) / (2*np.pi)

    return tiles(iht, numTilings, tile_dim*s, [tile_dim*a])


def q_hat(s, a):
    return weights[mytiles(s, a)].sum()


class Turtlebot3Burger(Robot):

    def __init__(self):
        super().__init__()

        self.forward_velocity = 0.05
        self.wheel_radius = 0.03
        self.body_width = 0.06

        self.time_step = int(self.getBasicTimeStep())

        self.left_wheel = self.getDevice("left motor")
        self.right_wheel = self.getDevice("right motor")

        self.left_wheel.setPosition(np.inf)
        self.right_wheel.setPosition(np.inf)

        self.lidar_sensor = self.getDevice("lidar")
        self.lidar_sensor.enable(self.time_step)
        # self.lidar_sensor.enablePointCloud()

        self.angular_velocities = np.array([-1.15, 0., 1.15])

    def actuate(self, action=None):

        angular_velocity = self.angular_velocities[action]

        left_speed = (
            self.forward_velocity - 0.5*self.body_width*angular_velocity
        ) / self.wheel_radius
        right_speed = (
            self.forward_velocity + 0.5*self.body_width*angular_velocity
        ) / self.wheel_radius

        self.left_wheel.setVelocity(left_speed)
        self.right_wheel.setVelocity(right_speed)

    def parse_custom_data(self):
        data = self.getCustomData().split(";")

        return {
            "coords": np.array(data[0].split(" ")).astype(float),
            "reward": float(data[1]),
            "done": bool(data[2])
        }

    def observe(self):
        custom_data = self.parse_custom_data()
        sensor_data = np.array(self.lidar_sensor.getRangeImage())
        sensor_data[sensor_data > 1.] = 1.
        state = np.concatenate(
            [
                sensor_data,
                custom_data["coords"]
            ]
        )

        return state, custom_data["reward"], custom_data["done"]

    def control(self):

        action_delay = 1.
        delay_steps = int(1e3*action_delay / self.time_step)

        while self.step(self.time_step) != -1:

            curr_state, reward, done = self.observe()

            # with np.printoptions(precision=3, suppress=True):
            # print(curr_state)

            if np.random.uniform() < EXPLORE_RATE:
                action = np.random.randint(0, self.angular_velocities.shape[0])
            else:
                action = np.array(
                    [
                        q_hat(curr_state, a) for a in range(
                            self.angular_velocities.shape[0]
                        )
                    ]
                ).argmax()

            self.actuate(action)

            while delay_steps > 0:
                self.step(self.time_step)
                next_state, reward, done = self.observe()

                if done:
                    state_tiles = mytiles(curr_state, action)
                    q_hat_next = np.max(
                        np.array(
                            [
                                q_hat(next_state, a) for a in range(
                                    self.angular_velocities.shape[0]
                                )
                            ]
                        )
                    )
                    weights[state_tiles] += stepSize * (
                        reward + DISCOUNT * q_hat_next - q_hat(
                            curr_state, action
                        )
                    )

                delay_steps -= 1

            delay_steps = int(1e3*action_delay / self.time_step)


if __name__ == "__main__":
    # create the Robot instance.
    robot = Turtlebot3Burger()
    robot.actuate(action=1)
    robot.control()
