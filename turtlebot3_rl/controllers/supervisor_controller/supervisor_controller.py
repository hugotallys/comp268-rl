"""supervisor_controller controller."""

import numpy as np

from controller import Supervisor

ROBOT_RADIUS = 0.1*np.sqrt(2) + 0.05
MAX_ITER = 500

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
        self.burger_node = self.getFromDef("WAFFLE")

        self.goal_position = [0.5, 0.5, 0.05]
        self.burger_position = np.array([0., 0., 0.])
        self.distance_to_goal = 0.

    def set_goal_position(self):
        idxs = np.random.randint(0, self.grid_x.shape[0], 2)
        x_coord = self.grid_x[idxs[0]]
        y_coord = self.grid_x[idxs[1]]
        self.goal_position = [x_coord, y_coord, 0.0]
        self.goal_node.getField(
            "translation"
        ).setSFVec3f(self.goal_position)

    def send_data(self, reward=None, done=""):
        self.burger_position = np.array(
            self.burger_node.getPosition()
        )
        gvec = (
            np.array(self.goal_position) - self.burger_position
        )
        self.distance_to_goal = np.linalg.norm(gvec)
        yaw = np.arctan2(gvec[1], gvec[0])
        R_d = 2**(np.linalg.norm(self.goal_position) / self.distance_to_goal)
        
        if reward is None:
            reward = R_d

        msg = f"{self.distance_to_goal} {yaw};{reward};{done}"
        
        self.burger_node.getField(
            "customData"
        ).setSFString(msg)

    def done(self):
        for i in range(2):
            if self.burger_position[i] + ROBOT_RADIUS > 1.:
                self.send_data(-10, "done")
                return True
            elif self.burger_position[i] - ROBOT_RADIUS < -1.:
                self.send_data(-10, "done")
                return True  
        if self.distance_to_goal < np.sqrt(2) * 0.125: # considerar raio do circilo circuiscrtio ???
            self.send_data(100, "done")
            return True
        self.send_data()
        return False

    def control(self):
        while self.step(self.time_step) != -1:
            if self.done():
                self.burger_node.getField(
                    "translation"
                ).setSFVec3f([0., 0., 0.05])
                # self.set_goal_position()

if __name__ == "__main__":

    robot = SimulationSurpervisor()  # create Supervisor instance
    # robot.set_goal_position()
    robot.control()
