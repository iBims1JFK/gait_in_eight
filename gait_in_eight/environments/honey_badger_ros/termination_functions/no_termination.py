import numpy as np


class NoTermination:
    def __init__(self, env):
        self.env = env
        self.power_history_length = 3

    def setup(self):
        self.power_history = np.zeros((self.power_history_length,))

    def should_terminate(self, obs):
        return False
