import numpy as np


class OnlySetupSampling:
    def __init__(self, env, probability=0.002):
        self.env = env
        self.probability = probability

    def setup(self):
        return True

    def step(self):
        return self.env.episode_step == 0
