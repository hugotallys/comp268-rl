import numpy as np

from controller import Supervisor  # type: ignore

ROBOT_RADIUS = 0.07


class Obstacle:

    def __init__(self, id):
        self.id = id
        self.x = np.random.uniform(-0.5, 0.5)
        self.y = np.random.uniform(-0.5, 0.5)
        self.z = 0.05
        self.radius = 0.025

    def get_wbo_str(self):
        with open("obstacle.wbo", "r") as wbo_file:
            return wbo_file.read().replace("DEF BOX_0", f"DEF BOX_{self.id}")

    def get_obstacle_position(self):
        return [self.x, self.y, self.z]


class SimulationSurpervisor(Supervisor):

    def __init__(self, n=0):
        super().__init__()

        self.time_step = int(self.getBasicTimeStep())

        self.root_node = self.getRoot()
        self.children_field = self.root_node.getField("children")
        self.children_field.importMFNode(-1, "goal.wbo")

        self.goal_node = self.getFromDef("GOAL")
        self.burger_node = self.getFromDef("DWROBOT")

        self.goal_position = [0.5, 0.5, 0]
        self.distance_to_goal = np.linalg.norm(self.goal_position)

        self.goal_node.getField(
            "translation"
        ).setSFVec3f(self.goal_position)

        self.burger_position = np.array([0., 0., 0.])

        self.cumul_reward = 0.

        self.obstacles = [Obstacle(id=i) for i in range(n)]

        for i in range(n):
            self.children_field.importMFNodeFromString(
                -1, self.obstacles[i].get_wbo_str()
            )
            box = self.getFromDef(f"BOX_{i}")
            box.getField(
                "translation"
            ).setSFVec3f(self.obstacles[i].get_obstacle_position())
            box.getField(
                "rotation"
            ).setSFRotation([0, 0, 1, np.random.uniform(0, 2*np.pi)])

    def send_data(self, reward=None, done=""):

        self.burger_position = self.burger_node.getField(
            "translation"
        ).getSFVec3f()

        gvec = (
            np.array(self.goal_position) - self.burger_position
        )

        self.distance_to_goal = np.linalg.norm(gvec)

        if reward is None:
            reward = -1e-3 * (1 + self.distance_to_goal)

        self.cumul_reward += reward

        msg = (
            f"{self.distance_to_goal} "
            f"{np.arctan2(gvec[1], gvec[0])};{reward};{done}"
        )

        self.burger_node.getField(
            "customData"
        ).setSFString(msg)

    def check_collision(self, x0, y0, x1, y1, d):
        return np.sqrt((x0 - x1)**2 + (y0 - y1)**2) < d

    def done(self):
        for obs in self.obstacles:
            obs_pos = obs.get_obstacle_position()
            collided = self.check_collision(
                x0=self.burger_position[0], y0=self.burger_position[1],
                x1=obs_pos[0], y1=obs_pos[1],
                d=ROBOT_RADIUS + obs.radius
            )
            if collided:
                self.send_data(-5, "done")
                self.cumul_reward = 0.
                return True

        for i in range(2):
            if self.burger_position[i] + ROBOT_RADIUS > 1:
                self.send_data(-5, "done")
                self.cumul_reward = 0.
                return True
            elif self.burger_position[i] - ROBOT_RADIUS < -1:
                self.send_data(-5, "done")
                self.cumul_reward = 0.
                return True

        if self.distance_to_goal - 0.5*ROBOT_RADIUS < 0.125:
            self.send_data(20, "done")
            self.cumul_reward = 0.
            return True
        self.send_data()
        return False

    def control(self):
        while self.step(self.time_step) != -1:
            if self.done():
                self.burger_node.getField(
                    "translation"
                ).setSFVec3f([0., 0., 0.05])
                self.burger_node.setVelocity([0.]*6)


if __name__ == "__main__":

    robot = SimulationSurpervisor(n=0)
    robot.control()
