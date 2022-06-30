"""supervisor_controller controller."""

import numpy as np
import time


from controller import Supervisor

ROBOT_RADIUS = 0.037
MAX_ITER = 500

class SimulationSurpervisor(Supervisor):

    def __init__(self):
        super().__init__()
        
        self.time_step = int(self.getBasicTimeStep())
        
        self.grid_x = np.arange(-.5 + 0.125, .5, 0.125)
        self.grid_y = np.arange(-.5 + 0.125, .5, 0.125)

        self.root_node = self.getRoot()
        self.children_field = self.root_node.getField("children")
        self.children_field.importMFNode(-1, "goal.wbo")

        self.goal_node = self.getFromDef("GOAL")
        self.burger_node = self.getFromDef("EPUCK")

        self.goal_position = [0.25, 0.25, 0]

        self.goal_node.getField(
            "translation"
        ).setSFVec3f(self.goal_position)

        self.burger_position = np.array([0., 0., 0.])
        self.burger_rotation = 0.
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

        self.burger_rotation = self.burger_node.getField("rotation").getSFRotation()
        
        z = self.burger_rotation[-2]
        a = self.burger_rotation[-1]
        
        self.burger_rotation = a * z

        self.burger_position = self.burger_node.getField("translation").getSFVec3f()
        gvec = (
            np.array(self.goal_position) - self.burger_position
        )
        self.distance_to_goal = np.linalg.norm(gvec)
        yaw = np.arctan2(gvec[1], gvec[0])
        R_d = 1 / (1 + self.distance_to_goal) # 2**(np.linalg.norm(self.goal_position) / self.distance_to_goal)

        R_r = 1.

        ymr = abs(yaw - self.burger_rotation)

        if ymr > np.pi/4:
            R_r = -1.

        if reward is None:
            reward = R_d * R_r

        # print("YAW=%.3f \t ROT=%.3f \t |YAW-ROT|=%.3f \t REW=%.2f" % (  (180/np.pi)*yaw, (180/np.pi)*self.burger_rotation, (180/np.pi)*ymr,  reward )) 

        msg = f"{self.distance_to_goal} {yaw};{reward};{done}"
        
        self.burger_node.getField(
            "customData"
        ).setSFString(msg)

    def done(self):
        for i in range(2):
            if self.burger_position[i] + ROBOT_RADIUS > 0.5:
                self.send_data(-5, "done")
                return True
            elif self.burger_position[i] - ROBOT_RADIUS < -0.5:
                self.send_data(-5, "done")
                return True  
        if self.distance_to_goal < np.sqrt(2) * 0.0625:
            self.send_data(10, "done")
            return True
        self.send_data()
        return False

    def control(self):
        while self.step(self.time_step) != -1:
            if self.done():
                self.burger_node.getField(
                    "translation"
                ).setSFVec3f([0., 0., 0.0])
                # self.set_goal_position()

if __name__ == "__main__":

    robot = SimulationSurpervisor()  # create Supervisor instance
    # robot.set_goal_position()
    robot.control()
