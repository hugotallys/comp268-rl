"""turtlebot_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from asyncio import current_task
import time
from turtle import distance
import numpy as np

from controller import Robot

from tiles3 import tiles, IHT, tileswrap


DISCOUNT = 0.9  # gamma
EXPLORE_RATE = 0.25  # epsilon

maxSize = 2**20
iht = IHT(maxSize)
numTilings = 32
stepSize = 0.1/numTilings

print(stepSize)

weights = np.zeros(shape=maxSize)


def mytiles(X, tile_dim=5.0):

    a = X[-1]
    X = X[:-1]
    
    a = tile_dim*a / 2
        
    for i in range(8):
        X[i] = (X[i] - 50.) / (2200 - 50)
    
    X[-2] =  (X[-2] + np.pi) / (2*np.pi)
        
    mask = [False for _ in range (9)]
    mask[-1] = tile_dim
    
    return tileswrap(iht, numTilings, tile_dim*X, mask, [a])


def q_hat(s, a):
    X = np.concatenate([s, np.array([a])])
    return weights[mytiles(X)].sum()


class Turtlebot3Burger(Robot):

    def __init__(self):
        super().__init__()

        self.forward_velocity = 0.05
        self.wheel_radius = 0.0205
        self.body_width = 0.052
        
        self.min_s = 9999
        self.max_s = 0

        self.time_step = int(self.getBasicTimeStep())

        self.left_wheel = self.getDevice("left wheel motor")
        self.right_wheel = self.getDevice("right wheel motor")

        self.left_wheel.setPosition(np.inf)
        self.right_wheel.setPosition(np.inf)
        
        self.distance_sensors = [self.getDevice("ps%d" % i) for i in range(8)]

        for d in self.distance_sensors:
            d.enable(self.time_step)

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
            "goal": np.array(data[0].split(" ")).astype(float),
            "reward": float(data[1]),
            "done": bool(data[2])
        }

    def observe(self, lidar_resolution=8):
        custom_data = self.parse_custom_data()
        sensor_data = np.array([d.getValue() for d in self.distance_sensors])
        
        '''min_s, max_s = sensor_data.min(), sensor_data.max()
        self.min_s = min_s if min_s < self.min_s else self.min_s
        self.max_s = max_s if max_s > self.max_s else self.max_s
        print(self.min_s, self.max_s)'''
        
        return np.concatenate(
            [
                sensor_data,
                custom_data["goal"]
            ]
        ), custom_data["reward"], custom_data["done"]

    def control(self):

        action_delay = 1.
        delay_steps = int(1e3*action_delay / self.time_step)
        
        while self.step(self.time_step) != -1:
            
            curr_state, reward, done = self.observe()
            
            print(reward)

            if np.random.uniform() < EXPLORE_RATE:
                action = np.random.randint(0, self.angular_velocities.shape[0])
            else:
                action = np.array([q_hat(curr_state, a) for a in range(self.angular_velocities.shape[0])]).argmax()
            
            self.actuate(action)
            
            while delay_steps > 0:
                self.step(self.time_step)
                next_state, reward, done = self.observe()
                
                if done:
                    # print("BAD REWARD!", reward)
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
                
                delay_steps -= 1
            
            delay_steps = int(1e3*action_delay / self.time_step)
            
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

if __name__ == "__main__":
    # create the Robot instance.
    robot = Turtlebot3Burger()
    robot.actuate(action=1)
    robot.control()
