import numpy as np

from controller import Robot  # type: ignore
from tile_coding import Tile


class DisplayGraph:

    def __init__(self, display, buffer_size):
        self.display = display

        self.width = self.display.getWidth()
        self.height = self.display.getHeight()

        self.buffer_size = buffer_size
        self.y_axis = np.zeros(shape=self.buffer_size)
        self.x_axis = np.linspace(0, self.width, self.buffer_size)
        self.pointer = 0

        self.moving_mean = 25

        self.update_plot(0., 1, 0.1)

    def update_plot(self, value, ep, er):

        if self.pointer < self.buffer_size - 1:
            self.pointer += 1
        else:
            self.y_axis = np.roll(self.y_axis, -1)

        value = self.map(value, -10, 25)
        self.y_axis[self.pointer] = value

        self.display.setColor(0)
        self.display.fillRectangle(0, 0, self.width, self.height)

        self.display.setFont("Lucida Console", 14, True)
        self.display.setColor(0xFFFFFF)
        self.display.drawText("Acumulated Reward x Epochs", 10, 10)

        self.display.setColor(0xFFFFFF)
        self.display.drawText("Epoch %d" % (ep,), self.width - 220, 10)

        self.display.setColor(0xFFFFFF)
        self.display.drawText(
            "Explore rate = %.2f" % (er,), self.width - 220, 30
        )

        zero_value = int(self.map(0, -10, 25))
        self.display.drawLine(
            0, self.height - zero_value,
            self.width, self.height - zero_value
        )

        y_axis_mean = self.runningMeanFast(self.y_axis)

        for i in range(1, self.pointer - self.moving_mean):
            self.display.setColor(0xFF00FF)
            x1 = int(self.x_axis[i])
            y1 = int(y_axis_mean[i])
            x0 = int(self.x_axis[i-1])
            y0 = int(y_axis_mean[i-1])
            self.display.drawLine(
                x0, self.height - y0,
                x1, self.height - y1
            )

    def map(self, value, y_min, y_max):
        return ((value - y_min) / (y_max - y_min)) * self.height

    def runningMeanFast(self, x):
        N = self.moving_mean
        return np.convolve(x, np.ones((N,))/N)[(N-1):]


class DWRobot(Robot):

    wheel_radius = 0.03
    body_diameter = 0.06
    action_delay = 1000
    tile = Tile(tiling_size=10, num_tilings=128, iht_size=2**22)
    weights = np.zeros(shape=tile.iht_size)
    ranges = [(0., 2.)] * 4 + [(0., 1.), (-np.pi, np.pi)]

    discount = 0.8
    explore = 0.1

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

        self.display = DisplayGraph(self.getDevice("display"), 200)

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
        epoch = 1
        cumul_reward = 0.
        delay_steps = int(self.action_delay / self.time_step)

        while self.step(self.time_step) != -1:

            curr_state, reward, done = self.observe()
            cumul_reward += reward

            if np.random.uniform() < self.explore:
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
                cumul_reward += reward

                if done:
                    if reward > 0:
                        self.explore = max(self.explore - 0.01, 0.01)
                    elif reward < 0:
                        self.explore = min(self.explore + 0.01, 0.10)

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
                        reward + self.discount * q_hat_next - self.q_hat(
                            curr_state, action
                        )
                    )
                    self.display.update_plot(cumul_reward, epoch, self.explore)
                    cumul_reward = 0.
                    epoch += 1

                delay_steps -= 1

            delay_steps = int(self.action_delay / self.time_step)


if __name__ == "__main__":
    # create the Robot instance.
    robot = DWRobot(lidar_pc=False)
    robot.control()
