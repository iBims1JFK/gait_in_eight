import numpy as np
import itertools

class CurriculumCommands:
    def __init__(self, env,
                 min_x_velocity=-1.0, max_x_velocity=1.0,
                 min_y_velocity=-1.0, max_y_velocity=1.0,
                 min_yaw_velocity=-0.1, max_yaw_velocity=0.1):
        self.env = env
        self.min_x_velocity = min_x_velocity
        self.max_x_velocity = max_x_velocity
        self.min_y_velocity = min_y_velocity
        self.max_y_velocity = max_y_velocity
        self.min_yaw_velocity = min_yaw_velocity
        self.max_yaw_velocity = max_yaw_velocity
        self.max_target_velocity = env.target_velocity
        self.target_velocity = env.target_velocity

        self.boundery = [0, 0]
        self.boundery_performance = [0, 0]
        self.boundery_size = np.pi / 5
        self.last_direction = 0
        self.level_up = False

        self.performance_history = np.zeros((400, 3))

    def get_next_command(self):
        if self.env.eval:
            pass
        else:
            self.performance_history = np.zeros((400, 3))
            self.level_up = False
            self.target_velocity = self.max_target_velocity

            if np.isclose(np.sum(self.boundery), 2 * np.pi):
                part = 2
            else:
                if np.isin(self.boundery, [0]).any():
                    part = np.random.choice([0, 1])
                else:
                    part = np.random.choice([0, 0, 1, 1, 2])
                if self.target_velocity > 0.25:
                    self.target_velocity = 0.25
            range = [0, 0]
            if part == 0:
                range = [-self.boundery[0] - self.boundery_size, -self.boundery[0]]
            elif part == 2:
                range = [-self.boundery[0], self.boundery[1]]
                if np.random.uniform(0, 1) > 0.1:
                    self.target_velocity = np.random.uniform(0.1, self.target_velocity)
                else:
                    self.target_velocity = 0
            elif part == 1:
                range = [self.boundery[1], self.boundery[1] + self.boundery_size]


            direction = np.random.uniform(range[0], range[1])
            self.last_direction = [direction, part]

            goal_x_velocity = -np.cos(direction)
            goal_y_velocity = -np.sin(direction)
            goal_yaw_velocity = 0

            if not np.linalg.norm(np.array([goal_x_velocity, goal_y_velocity])) == 0:
                scaled_vector = self.target_velocity * np.array([goal_x_velocity, goal_y_velocity]) / np.linalg.norm(np.array([goal_x_velocity, goal_y_velocity]))
            else:
                scaled_vector = np.array([0.0, 0.0])
        return *scaled_vector, goal_yaw_velocity
    
    def setup(self):
        return

    def step(self, obs, reward, absorbing, info):
        if not self.env.mode == "test" and not self.env.eval:
            last_direction = self.last_direction[0]
            rotation = self.rotation(-last_direction - np.pi)
            
            current_yaw_vel = self.env.data.qvel[5]
            current_global_linear_velocity = self.env.data.qvel[:3]
            
            current_local_linear_velocity = self.env.orientation_quat_inv.apply(current_global_linear_velocity)
            current_local_target_linear_velocity = rotation @ current_local_linear_velocity[:2]

            self.performance_history = np.roll(self.performance_history, 1, axis=0)
            self.performance_history[0] = np.array([current_local_target_linear_velocity[0], current_local_target_linear_velocity[1], current_yaw_vel])
            avg = np.mean(self.performance_history, axis=0)
            if not self.last_direction[1] == 2 and avg[0] >= self.target_velocity * 0.95 and np.abs(avg[1]) < 0.05 and np.abs(avg[2]) < 0.05 and not self.level_up:
                self.boundery[self.last_direction[1]] += self.boundery_size
                self.boundery[self.last_direction[1]] = np.clip(self.boundery[self.last_direction[1]], 0, 2 * np.pi - self.boundery[1 - self.last_direction[1]])
                self.level_up = True
                print("Level up!", self.boundery, avg)
            
            if absorbing:
                info[f"Curriculum/Boundery/left"] = self.boundery[0] / np.pi
                info[f"Curriculum/Boundery/right"] = self.boundery[1] / np.pi
                info[f"Curriculum/Boundery/total"] = np.sum(self.boundery) / np.pi
                info[f"Curriculum/perf_td/avg_x"] = avg[0]
                info[f"Curriculum/perf_td/avg_y"] = avg[1]
                info[f"Curriculum/perf_td/avg_yaw"] = avg[2]
 


    def rotation(self, angle):
        return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])