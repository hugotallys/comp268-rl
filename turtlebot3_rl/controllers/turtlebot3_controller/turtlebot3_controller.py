"""turtlebot_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
import time
import numpy as np

from controller import Robot


class Turtlebot3Burger(Robot):

    def __init__(self):
        super().__init__()

        self.forward_velocity = 0.1
        self.wheel_radius = .033
        self.body_width = .160

        self.time_step = int(self.getBasicTimeStep())

        self.left_wheel = self.getDevice("left wheel motor")
        self.right_wheel = self.getDevice("right wheel motor")

        self.left_wheel.setPosition(np.inf)
        self.right_wheel.setPosition(np.inf)

        self.lidar_sensor = self.getDevice("LDS-01")
        self.lidar_sensor.enable(self.time_step)

        self.angular_velocities = np.array([-1.15, -.75, 0., .75, 1.15])
        # self.lidar_sensor.enablePointCloud()

    def actuate(self, action, random=False):

        if random:
            angular_velocity = self.angular_velocities[np.random.randint(0, self.angular_velocities.shape[0])]
        else:
            angular_velocity = self.angular_velocities[action]

        left_speed = (
            self.forward_velocity - 0.5*self.body_width*angular_velocity
        ) / self.wheel_radius
        right_speed = (
            self.forward_velocity + 0.5*self.body_width*angular_velocity
        ) / self.wheel_radius

        self.left_wheel.setVelocity(left_speed)
        self.right_wheel.setVelocity(right_speed)

    def get_state(self, lidar_resolution=8):
        step = int(
            self.lidar_sensor.getHorizontalResolution() / lidar_resolution
        )
        return np.concatenate(
            [
                self.lidar_sensor.getRangeImage()[::step],
                np.array(self.getCustomData().split(" ")).astype(float)
            ]
        )

    def control(self):

        init_time = time.time()

        while self.step(self.time_step) != -1:
            now = time.time()

            if (now - init_time) > 2.:
                self.actuate(None, random=True)
                init_time = now

if __name__ == "__main__":
    # create the Robot instance.
    robot = Turtlebot3Burger()
    robot.actuate(action=2)
    robot.control()

# Main loop:
# - perform simulation steps until Webots is stopping the controller

# Enter here exit cleanup code.
