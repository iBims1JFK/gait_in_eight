import numpy as np


class TrunkCollisionAndPowerTermination:
    def __init__(self, env):
        self.env = env
        self.power_history_length = 3

    def setup(self):
        self.power_history = np.zeros((self.power_history_length,))

    def should_terminate(self, obs):
        ids = self.env.collision_groups["trunk"]

        trunk_collision = False
        for con_i in range(0, self.env.data.ncon):
            con = self.env.data.contact[con_i]
            if con.geom1 in ids or con.geom2 in ids:
                trunk_collision = True
                break

        power = np.sum(abs(self.env.current_torques) * abs(self.env.data.qvel[6:]))
        self.power_history = np.roll(self.power_history, shift=-1)
        self.power_history[-1] = power
        power_limit_reached = np.all(self.power_history > self.env.power_limit_watt)

        return trunk_collision or power_limit_reached
