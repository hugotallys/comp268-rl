"""turtlebot_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
import time
import numpy as np

from controller import Robot

from tiles3 import tiles, IHT


DISCOUNT = 0.86  # gamma
EXPLORE_RATE = 0.01  # epsilon
EPISODES = 50
MAX_ITER = 250

maxSize = 2^20
iht = IHT(maxSize)
numTilings = 8
stepSize = 0.1/numTilings

weights = np.zeros(shape=maxSize)


def mytiles(X, tile_dim=10.0):
    X[-2] = X[-2] / (2*np.sqrt(2))
    X[-1] = (np.pi + X[-1]) / (2*np.pi)
    return tiles(iht, numTilings, tile_dim*X)


def q_hat(s, a):
    X = np.concatenate([s, np.array([a])])
    return weights[mytiles(X)].sum()


class Turtlebot3Burger(Robot):

    def __init__(self):
        super().__init__()

        self.forward_velocity = 0.1
        self.wheel_radius = 0.05
        self.body_width = 0.2

        self.time_step = int(self.getBasicTimeStep())

        self.left_wheel = self.getDevice("left wheel motor")
        self.right_wheel = self.getDevice("right wheel motor")

        self.left_wheel.setPosition(np.inf)
        self.right_wheel.setPosition(np.inf)

        self.lidar_sensor = self.getDevice("LDS-01")
        self.lidar_sensor.enable(self.time_step)

        self.angular_velocities = np.array([-1.15, -.75, 0., .75, 1.15])
        self.lidar_sensor.enablePointCloud()

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
            "goal": np.array(data[0].split(" ")).astype(float),
            "reward": float(data[1]),
            "done": bool(data[2])
        }

    def observe(self, lidar_resolution=8):
        step = int(
            self.lidar_sensor.getHorizontalResolution() / lidar_resolution
        )
        custom_data = self.parse_custom_data()
        lidar_data = np.array(self.lidar_sensor.getRangeImage())
        lidar_data[lidar_data > 2] = np.sqrt(2)*2.
        return np.concatenate(
            [
                lidar_data,
                custom_data["goal"]
            ]
        ), custom_data["reward"], custom_data["done"]

    def control(self):

        init_time = time.time()

        while self.step(self.time_step) != -1:
            
            now = time.time()

            curr_state, reward, done = self.observe()

            print(reward)

            if np.random.uniform() < EXPLORE_RATE:
                action = np.random.randint(0, self.angular_velocities.shape[0])
            else:
                action = np.array([q_hat(curr_state, a) for a in range(self.angular_velocities.shape[0])]).argmax()
            
            if now - init_time > .5:
                self.actuate(action)
                init_time = now
            
            self.step(self.time_step)
            
            next_state, reward, done = self.observe()
            state_tiles = mytiles(curr_state, action)
            q_hat_next = np.max(
                np.array(
                    [
                        q_hat(next_state, a) for a in range(self.angular_velocities.shape[0])
                    ]
                )
            )
            weights[state_tiles] += stepSize * (
                reward + DISCOUNT * q_hat_next - q_hat(curr_state, action)
            )
            if done:
                pass #TODO

if __name__ == "__main__":
    # create the Robot instance.
    robot = Turtlebot3Burger()
    robot.actuate(action=2)
    robot.control()

# Main loop:
# - perform simulation steps until Webots is stopping the controller

# Enter here exit cleanup code.
