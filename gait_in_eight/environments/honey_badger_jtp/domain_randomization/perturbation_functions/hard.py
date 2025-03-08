import numpy as np


class HardDomainPerturbation:
    def __init__(self, env,
                 push_velocity_x_min=-1.0, push_velocity_x_max=1.0,
                 push_velocity_y_min=-1.0, push_velocity_y_max=1.0,
                 push_velocity_z_min=-1.0, push_velocity_z_max=1.0,
                 add_chance=0.5, additive_multiplier=1.5):
        self.env = env
        self.push_velocity_x_min = push_velocity_x_min
        self.push_velocity_x_max = push_velocity_x_max
        self.push_velocity_y_min = push_velocity_y_min
        self.push_velocity_y_max = push_velocity_y_max
        self.push_velocity_z_min = push_velocity_z_min
        self.push_velocity_z_max = push_velocity_z_max
        self.add_chance = add_chance
        self.additive_multiplier = additive_multiplier

    def sample(self):
        self.sampled_push_velocity_x = self.env.np_rng.uniform(self.push_velocity_x_min, self.push_velocity_x_max)
        self.sampled_push_velocity_y = self.env.np_rng.uniform(self.push_velocity_y_min, self.push_velocity_y_max)
        self.sampled_push_velocity_z = self.env.np_rng.uniform(self.push_velocity_z_min, self.push_velocity_z_max)
        if self.env.np_rng.uniform() < self.add_chance:
            self.env.data.qvel[0:3] += np.array([self.sampled_push_velocity_x, self.sampled_push_velocity_y, self.sampled_push_velocity_z])
        else:
            self.env.data.qvel[0:3] = np.array([self.sampled_push_velocity_x, self.sampled_push_velocity_y, self.sampled_push_velocity_z]) * self.additive_multiplier
