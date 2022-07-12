import numpy as np

from controller import Robot  # type: ignore
from tile_coding import Tile

DISCOUNT = 0.96
EXPLORE_RATE = 0.15


class DWRobot(Robot):

    wheel_radius = 0.03
    body_diameter = 0.06
    action_delay = 1000
    tile = Tile(tiling_size=3, num_tilings=64, iht_size=2**22)
    weights = np.zeros(shape=tile.iht_size)
    ranges = [(0., 2.)] * 4 + [(0., 1.), (-np.pi, np.pi)]

    def __init__(self, lin_vel=0.05, ang_vel=0.25*np.pi, lidar_pc=False):
        super().__init__()

        self.lin_vel = lin_vel
        self.ang_vel = ang_vel

        self.time_step = int(self.getBasicTimeStep())

        self.left_wheel = self.getDevice("left motor")
        self.right_wheel = self.getDevice("right motor")

        self.left_wheel.setPosition(np.inf)
        self.right_wheel.setPosition(np.inf)

        self.lidar_sensor = self.getDevice("lidar")
        self.lidar_sensor.enable(self.time_step)

        if lidar_pc:
            self.lidar_sensor.enablePointCloud()

        self.actions = [
            (self.lin_vel, 0.),
            (0, 8*self.ang_vel),
            (0., 2*self.ang_vel),
            (0., -2*self.ang_vel)
        ]

        self.ranges.append((0, len(self.actions)-1))

    def actuate(self, action=None):

        v, w = self.actions[action]

        left_speed = (
            v - 0.5*self.body_diameter*w
        ) / self.wheel_radius
        right_speed = (
            v + 0.5*self.body_diameter*w
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

    def q_hat(self, s, a):
        sa = np.concatenate((s, np.array([a])))
        return self.weights[self.tile.tile(sa, self.ranges)].sum()

    def control(self):
        delay_steps = int(self.action_delay / self.time_step)

        while self.step(self.time_step) != -1:

            curr_state, reward, done = self.observe()

            if np.random.uniform() < EXPLORE_RATE:
                action = np.random.randint(0, len(self.actions))
            else:
                action = np.array(
                    [
                        self.q_hat(curr_state, a) for a in range(
                            len(self.actions)
                        )
                    ]
                ).argmax()

            self.actuate(action)

            while delay_steps > 0:
                self.step(self.time_step)
                next_state, reward, done = self.observe()

                if done:
                    state_action = np.concatenate(
                        (curr_state, np.array([action]))
                    )
                    state_tiles = self.tile.tile(state_action, self.ranges)
                    q_hat_next = np.max(
                        np.array(
                            [
                                self.q_hat(next_state, a) for a in range(
                                    len(self.actions)
                                )
                            ]
                        )
                    )
                    self.weights[state_tiles] += self.tile.step * (
                        reward + DISCOUNT * q_hat_next - self.q_hat(
                            curr_state, action
                        )
                    )

                delay_steps -= 1

            delay_steps = int(self.action_delay / self.time_step)


if __name__ == "__main__":
    # create the Robot instance.
    robot = DWRobot(lidar_pc=False)
    robot.control()
