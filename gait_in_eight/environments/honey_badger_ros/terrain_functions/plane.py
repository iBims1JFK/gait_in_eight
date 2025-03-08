import numpy as np


class PlaneTerrainGeneration:
    def __init__(self, env):
        self.env = env
        self.xml_file_name = "plane.xml"
        self.center_height = 0.0

    def get_height_samples(self):
        return np.array([self.env.data.qpos[2] - self.center_height])
    
    def step(self, obs, reward, absorbing, info):
        return
    
    def info(self, info):
        return info
    
    def sample(self):
        return
