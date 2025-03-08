import numpy as np


class AverageSmoothing:
    def __init__(self, env, history_length=5):
        self.env = env
        self.step_count = 0
        self.history_length = history_length


    def init(self):
        self.action_space = self.env.model.nu
        self.action_history = np.repeat(self.env.nominal_joint_positions, self.history_length).reshape(self.history_length, self.action_space)
    
    def step(self, action):
        self.action_history[self.step_count % self.history_length] = action
        self.step_count += 1
        return np.mean(self.action_history, axis=0)

