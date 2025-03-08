import numpy as np
from OneEuroFilter import OneEuroFilter as OEF
import time

class OneEuroFilter:
    def __init__(self, env):
        self.env = env
        self.action_space = 12
        self.filter_config =  {
            'freq': self.env.control_frequency_hz,       # Hz
            'mincutoff': 2.5,  # Hz
            'beta': 0.1,       
            'dcutoff': 100.0    
        }
        self.init_time = time.perf_counter()
        self.filter = [OEF(**self.filter_config) for _ in range(self.env.model.nu)]

    def init(self):
        self.init_time = time.perf_counter()
        self.filter = [OEF(**self.filter_config) for _ in range(self.env.model.nu)]

    def step(self, action):
        filtered_action = np.empty(self.action_space)
        current_time = time.perf_counter() - self.init_time
        for i in range(self.action_space):
            filtered_action[i] = self.filter[i](action[i], current_time)
        return filtered_action
            