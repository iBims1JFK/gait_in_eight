import numpy as np
import itertools
import mujoco
from gait_in_eight.environments import observation_indices_wo_feet_cpg as obs_idx


class YawEvaluation:
    def __init__(self, env):
        self.env = env
        self.target_velocity = env.target_velocity
        self.n_height_field = (40, 40)

        self.eval_step = 0
        self.terrain_id = 0

        self.last_dir = 0


    def get_next_command(self):
        if self.env.eval:
            r = np.random.uniform(0, 1)
            bound =  np.pi / 3
            if r < 0.5:
                direction = np.random.uniform(-bound, bound)
            else:
                direction = np.random.uniform(np.pi - bound, np.pi + bound)
            self.last_dir = direction
            goal_yaw_velocity = np.random.uniform(-0.3, 0.3)
            vel = np.random.uniform(0.0, self.target_velocity)

            goal_x_velocity = np.cos(direction)
            goal_y_velocity = np.sin(direction)

            if not np.linalg.norm(np.array([goal_x_velocity, goal_y_velocity])) == 0:
                scaled_vector = vel * np.array([goal_x_velocity, goal_y_velocity]) / np.linalg.norm(np.array([goal_x_velocity, goal_y_velocity]))
            else:
                scaled_vector = np.array([0.0, 0.0])
            
            return *scaled_vector, goal_yaw_velocity
    def setup(self):
        return

    def step(self, obs, reward, absorbing, info):
        current_yaw_vel = self.env.data.qvel[5]
        current_global_linear_velocity = self.env.data.qvel[:3]
        current_local_linear_velocity = self.env.orientation_quat_inv.apply(current_global_linear_velocity)
        goal_vel = obs[obs_idx.GOAL_VELOCITIES]
        vel_mae = np.sum(np.abs(goal_vel[:2] - current_local_linear_velocity[:2]))

        vel_x_mae = np.abs(goal_vel[0] - current_local_linear_velocity[0])
        vel_y_mae = np.abs(goal_vel[1] - current_local_linear_velocity[1])
        vel_yaw_mae = np.abs(goal_vel[2] - current_yaw_vel)

        rotation_inv = np.array([[np.cos(-self.last_dir), -np.sin(-self.last_dir)], [np.sin(-self.last_dir), np.cos(-self.last_dir)]])
        current_target_vel = rotation_inv @ current_local_linear_velocity[:2]
        if self.terrain_id == 1:
            info["eval/rough/velocity_mae"] = vel_mae
            info["eval/rough/velocity_x_mae"] = vel_x_mae
            info["eval/rough/velocity_y_mae"] = vel_y_mae
            info["eval/flat/velocity_yaw_mae"] = vel_yaw_mae

        elif self.terrain_id == 0:
            info["eval/flat/velocity_mae"] = vel_mae
            info["eval/flat/velocity_x_mae"] = vel_x_mae
            info["eval/flat/velocity_y_mae"] = vel_y_mae
            info["eval/flat/velocity_yaw_mae"] = vel_yaw_mae

    

    def change_terrain(self):
        self.terrain_id = 1 - self.terrain_id
        if self.terrain_id == 1:
            nrows, ncols = self.n_height_field
            heights = np.random.rand(nrows, ncols)
            heights = heights.reshape((nrows, ncols))
            center_h = int((nrows - 1) / 2)
            center_w = int((ncols - 1) / 2)
            heights[center_h, center_w] = 0.1
            pad = 5
            for i in range(pad):
                for j in range(pad):
                    heights[int(center_h + i - np.floor(pad / 2)), int(center_w + j - np.floor(pad / 2))] = 0

            height_data = heights.flatten().tolist()
            self.env.model.hfield_data = height_data
        elif self.terrain_id == 0:
            self.env.model.hfield_data = np.zeros(self.n_height_field).flatten().tolist()

        if self.env.viewer:
            mujoco.mjr_uploadHField(self.env.viewer.model, self.env.viewer.context, 0)