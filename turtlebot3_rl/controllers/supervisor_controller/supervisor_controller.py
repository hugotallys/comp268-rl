"""supervisor_controller controller."""

import time
from urllib import robotparser
import numpy as np

from controller import Supervisor


class SimulationSurpervisor(Supervisor):

    def __init__(self):
        super().__init__()

        self.time_step = int(self.getBasicTimeStep())

        self.grid_x = np.arange(-1. + 0.25, 1., 0.25)
        self.grid_y = np.arange(-1. + 0.25, 1., 0.25)

        self.root_node = self.getRoot()
        self.children_field = self.root_node.getField("children")
        self.children_field.importMFNode(-1, "goal.wbo")

        self.goal_node = self.getFromDef("GOAL")
        self.burger_node = self.getFromDef("BURGER")

        self.goal_position = [0.5, 0.5, 0.]

    def set_goal_position(self):
        idxs = np.random.randint(0, self.grid_x.shape[0], 2)
        x_coord = self.grid_x[idxs[0]]
        y_coord = self.grid_x[idxs[1]]
        self.goal_position = [x_coord, y_coord, 0.0]
        self.goal_node.getField(
            "translation"
        ).setSFVec3f(self.goal_position)

    def send_goal_data(self):
        gvec = (
            np.array(self.goal_position) - np.array(
                self.burger_node.getPosition()
            )
        )
        self.distance_to_goal = np.linalg.norm(gvec)
        self.burger_node.getField(
            "customData"
        ).setSFString(
            f"{self.distance_to_goal} {np.arctan2(gvec[1], gvec[0])}"
        )

    def done(self):
        contact_points = len(self.burger_node.getContactPoints())
        if contact_points or self.distance_to_goal < 0.125: # considerar raio do circilo circuiscrtio ???
            return True
        return False

    def control(self):
        reset_counter = 0
        while self.step(self.time_step) != -1:
            self.send_goal_data()

            if self.done() and not reset_counter:
                self.burger_node.getField(
                    "translation"
                ).setSFVec3f([0., 0., 0.])
                # self.set_goal_position()
                while reset_counter < 5:
                    reset_counter += 1
                reset_counter = 0


if __name__ == "__main__":

    robot = SimulationSurpervisor()  # create Supervisor instance
    # robot.set_goal_position()
    robot.control()
