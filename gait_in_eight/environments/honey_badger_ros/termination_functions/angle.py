import numpy as np


class AngleTermination:
    def __init__(self, env):
        self.env = env
        self.limit = 0.25 * np.pi

    def setup(self):
        pass

    def should_terminate(self, obs):
        o = self.env.orientation_quat.apply(np.array([1.0, 0.0, 0.0]))
        m = np.sqrt(np.sum(o ** 2))
        theta = np.arccos(o[2] / m) - np.pi/2
        limit = np.deg2rad(30)
        if not -limit < theta < limit:
            return True 
        angular_vel = np.linalg.norm(self.env.data.qvel[3:6])
        return angular_vel > 4

