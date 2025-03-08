import numpy as np
import itertools
import mujoco
from gait_in_eight.environments import observation_indices_wo_feet_cpg as obs_idx


class DefaultEvaluation:
    def __init__(self, env):
        self.env = env
        self.target_velocity = env.target_velocity
        self.n_height_field = (40, 40)

        self.eval_step = 0
        self.terrain_id = 0

        target_directions = np.arange(-np.pi, np.pi, np.pi / 4)
        target_velocities = np.array([self.target_velocity / 2, self.target_velocity])
        self.targets = np.asarray(list(itertools.product(target_directions, target_velocities)))


    def get_next_command(self):
        if self.env.eval:
            if self.eval_step == 0:
                # self.change_terrain()
                pass

            if self.targets.shape[0] == self.eval_step:
                vel = 0
                dir = 0
            else:
                dir = self.targets[self.eval_step][0]
                vel = self.targets[self.eval_step][1]

            goal_x_velocity = np.cos(dir)
            goal_y_velocity = np.sin(dir)

            if not np.linalg.norm(np.array([goal_x_velocity, goal_y_velocity])) == 0:
                scaled_vector = vel * np.array([goal_x_velocity, goal_y_velocity]) / np.linalg.norm(np.array([goal_x_velocity, goal_y_velocity]))
            else:
                scaled_vector = np.array([0.0, 0.0])
            
            if self.targets.shape[0] > self.eval_step:
                self.eval_step = self.eval_step + 1
            else:
                self.eval_step = 0

            return *scaled_vector, 0
    def setup(self):
        return

    def step(self, obs, reward, absorbing, info):
        current_yaw_vel = self.env.data.qvel[5]
        current_global_linear_velocity = self.env.data.qvel[:3]
        current_local_linear_velocity = self.env.orientation_quat_inv.apply(current_global_linear_velocity)
        current_local_linear_velocity = current_local_linear_velocity[:2]
        target_vel = 1
        dir_code = []
        dir = 0

        if (self.eval_step - 1) == -1:
            dir_code.append("standing")
            goal_vel = np.array([0, 0])
            dir_eval = 1 - np.clip(np.linalg.norm(goal_vel - current_local_linear_velocity) / 1, 0, 1)

        else:
            target_vel = self.targets[self.eval_step - 1][1]

            dir = self.targets[self.eval_step - 1][0]

            goal_x_velocity = np.cos(dir)
            goal_y_velocity = np.sin(dir)
            goal_vel = target_vel * np.array([goal_x_velocity, goal_y_velocity]) / np.linalg.norm(np.array([goal_x_velocity, goal_y_velocity]))
            if np.isclose(goal_x_velocity, 0):
                # dir_code.append("straight")
                pass
            elif goal_x_velocity > 0:
                dir_code.append("N")
            else:
                dir_code.append("S")
            
            if np.isclose(goal_y_velocity, 0):
                # dir_code.append("straight")
                pass
            elif goal_y_velocity > 0:
                dir_code.append("W")
            else:
                dir_code.append("E")

        
            if (self.eval_step - 1) % 2 == 0:
                dir_code.append("half")
            else:
                dir_code.append("full")
            dir_eval = 1 - np.clip(np.linalg.norm(goal_vel - current_local_linear_velocity) / np.linalg.norm(goal_vel), 0, 1)
            # print(dir_eval, goal_vel, current_vel, goal_vel - current_vel, np.linalg.norm(goal_vel - current_vel))

        # vel_mae = np.linalg.norm(goal_vel - current_vel)
        vel_mae = np.sum(np.abs(goal_vel - current_local_linear_velocity))

        vel_x_mae = np.abs(goal_vel[0] - current_local_linear_velocity[0])
        vel_y_mae = np.abs(goal_vel[1] - current_local_linear_velocity[1])

        # print("current_vel", current_vel[0])
        full_dir_code = "_".join(dir_code)
        # if full_dir_code == "N_W_full":
        #     print(dir_eval, goal_vel, current_local_linear_velocity, goal_vel - current_local_linear_velocity, np.linalg.norm(goal_vel - current_local_linear_velocity))
        # print(full_dir_code, self.terrain_id)

        rotation_inv = np.array([[np.cos(-dir), -np.sin(-dir)], [np.sin(-dir), np.cos(-dir)]])
        current_target_vel = rotation_inv @ current_local_linear_velocity
        if self.terrain_id == 1:
            info["eval/rough/velocity_mae"] = vel_mae
            info["eval/rough/velocity_x_mae"] = vel_x_mae
            info["eval/rough/velocity_y_mae"] = vel_y_mae
        elif self.terrain_id == 0:
            info["eval/flat/velocity_mae"] = vel_mae
            info["eval/flat/velocity_x_mae"] = vel_x_mae
            info["eval/flat/velocity_y_mae"] = vel_y_mae
            # info[f"eval/flat/{full_dir_code}"] = vel_mae / (target_vel * 4)
            # info[f"eval/flat/{full_dir_code}/score"] = dir_eval
            # info[f"eval/flat/{full_dir_code}/vel1"] = current_target_vel[0]
            info[f"eval/flat/{full_dir_code}/x"] = current_target_vel[0]
            info[f"eval/flat/{full_dir_code}/y"] = current_target_vel[1]
            info[f"eval/flat/{full_dir_code}/yaw"] = current_yaw_vel
    

    def change_terrain(self):
        self.terrain_id = 1 - self.terrain_id
        if self.terrain_id == 1:
            nrows, ncols = self.n_height_field
            heights = np.random.rand(nrows, ncols)
            heights = heights.reshape((nrows, ncols))
            center_h = int((nrows - 1) / 2)
            center_w = int((ncols - 1) / 2)
            # heights[center_h, center_w] = 1
            # pad = 5
            # for i in range(pad):
            #     for j in range(pad):
            #         heights[int(center_h + i - np.floor(pad / 2)), int(center_w + j - np.floor(pad / 2))] = 0

            height_data = heights.flatten().tolist()
            self.env.model.hfield_data = height_data
        elif self.terrain_id == 0:
            self.env.model.hfield_data = np.zeros(self.n_height_field).flatten().tolist()

        if self.env.viewer:
            mujoco.mjr_uploadHField(self.env.viewer.model, self.env.viewer.context, 0)