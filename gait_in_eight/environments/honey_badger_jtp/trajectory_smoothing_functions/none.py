import numpy as np


class NoSmoothing:
    def __init__(self, env):
        self.env = env
    def init(self):
        pass
    
    def step(self, action):
        return action

