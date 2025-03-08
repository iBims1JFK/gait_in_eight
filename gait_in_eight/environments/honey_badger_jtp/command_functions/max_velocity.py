import numpy as np


class MaxVelocityCommands:
    def __init__(self, env,
                 min_x_velocity=-1.0, max_x_velocity=1.0,
                 min_y_velocity=-1.0, max_y_velocity=1.0,
                 min_yaw_velocity=-1.0, max_yaw_velocity=1.0):
        self.env = env
        self.min_x_velocity = min_x_velocity
        self.max_x_velocity = max_x_velocity
        self.min_y_velocity = min_y_velocity
        self.max_y_velocity = max_y_velocity
        self.min_yaw_velocity = min_yaw_velocity
        self.max_yaw_velocity = max_yaw_velocity
        self.target_velocity = env.target_velocity
        self.achieved_speed = 0
        self.current_target_velocity = 0.25
        self.performance_history = np.zeros(self.env.control_frequency_hz)
        self.level_up = False

    def get_next_command(self):
        if self.env.episode_step == 0:
            self.level_up = False
            self.performance_history = np.zeros(self.env.control_frequency_hz)
            goal_x_velocity = self.current_target_velocity
            goal_y_velocity = 0
            goal_yaw_velocity = 0

            return goal_x_velocity, goal_y_velocity, goal_yaw_velocity
        else:
            return 0.1, 0, 0

    def setup(self):
        return

    def step(self, obs, reward, absorbing, info):
        global_current_velocity = self.env.data.qvel[:3]
        x_current_velocity = self.env.orientation_quat_inv.apply(global_current_velocity)[0]
        self.performance_history = np.roll(self.performance_history, 1)
        self.performance_history[0] = x_current_velocity
        avg = np.mean(self.performance_history)

        if avg > self.achieved_speed and not self.level_up:
            self.achieved_speed = avg

            if self.current_target_velocity - self.achieved_speed < 0.1:
                self.current_target_velocity = self.current_target_velocity + 0.1
                self.current_target_velocity = np.clip(self.current_target_velocity, 0, self.target_velocity)
                self.performance_history = np.zeros(self.env.control_frequency_hz)
                self.level_up = True
                print("New target velocity: ", self.current_target_velocity)
        if absorbing:
            info["maxspeed/max_achieved_speed"] = self.achieved_speed
            info["maxspeed/current_speed"] = avg
            info["maxspeed/target_speed"] = self.current_target_velocity