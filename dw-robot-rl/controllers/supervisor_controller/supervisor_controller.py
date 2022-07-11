import numpy as np

from controller import Supervisor  # type: ignore

ROBOT_RADIUS = 0.07


class SimulationSurpervisor(Supervisor):

    def __init__(self):
        super().__init__()

        self.time_step = int(self.getBasicTimeStep())

        self.root_node = self.getRoot()
        self.children_field = self.root_node.getField("children")
        self.children_field.importMFNode(-1, "goal.wbo")

        self.goal_node = self.getFromDef("GOAL")
        self.burger_node = self.getFromDef("DWROBOT")

        self.goal_position = [0.25, 0.25, 0]
        self.distance_to_goal = np.linalg.norm(self.goal_position)

        self.goal_node.getField(
            "translation"
        ).setSFVec3f(self.goal_position)

        self.burger_position = np.array([0., 0., 0.])

        self.cumul_reward = 0.

    def send_data(self, reward=None, done=""):

        self.burger_position = self.burger_node.getField(
            "translation"
        ).getSFVec3f()

        gvec = (
            np.array(self.goal_position) - self.burger_position
        )

        self.distance_to_goal = np.linalg.norm(gvec)

        if reward is None:
            reward = 1e-3 / (1 + self.distance_to_goal)

        self.cumul_reward += reward

        vels = self.burger_node.getVelocity()

        msg = (
            f"{vels[0]} {vels[1]} {vels[-1]} "
            f"{self.distance_to_goal} "
            f"{np.arctan2(gvec[1], gvec[0])};{reward};{done}"
        )

        self.burger_node.getField(
            "customData"
        ).setSFString(msg)

    def done(self):
        for i in range(2):
            if self.burger_position[i] + ROBOT_RADIUS > 0.5:
                self.send_data(-5, "done")
                self.cumul_reward = 0.
                return True
            elif self.burger_position[i] - ROBOT_RADIUS < -0.5:
                self.send_data(-5, "done")
                self.cumul_reward = 0.
                return True
        if self.distance_to_goal - 0.5*ROBOT_RADIUS < 0.0625:
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

    robot = SimulationSurpervisor()
    robot.control()
