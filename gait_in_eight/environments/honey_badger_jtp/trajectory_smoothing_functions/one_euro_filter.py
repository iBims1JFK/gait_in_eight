import numpy as np
from OneEuroFilter import OneEuroFilter as OEF

class OneEuroFilter:
    def __init__(self, env):
        self.env = env
        self.action_space = 12
        self.frequency = self.env.control_frequency_hz
        self.filter_config =  {
            'freq': self.frequency,       # Hz
            'mincutoff': 2.5,  # Hz
            'beta': 0.1,       
            'dcutoff': 100.0    
        }
        # {
        #     'freq': self.frequency,       # Hz
        #     'mincutoff': 5.0,  # Hz
        #     'beta': 0.2,       
        #     'dcutoff': 10.0    
        # }
        self.i = 0
        self.filter = [OEF(**self.filter_config) for _ in range(self.env.model.nu)]

    def init(self):
        self.i = 0
        self.filter = [OEF(**self.filter_config) for _ in range(self.env.model.nu)]

    def step(self, action):
        filtered_action = np.empty(self.action_space)
        for i in range(self.action_space):
            filtered_action[i] = self.filter[i](action[i], self.i * 1 / self.frequency)
        self.i += 1
        return filtered_action
            