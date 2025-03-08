import numpy as np


class TorqueControl:
    def __init__(self, env, control_frequency_hz=50, scaling_factor=4.0):
        self.env = env
        self.control_frequency_hz = control_frequency_hz
        self.scaling_factor = scaling_factor
        self.motor_strength_factor = np.ones(12)

    def process_action(self, action):
        torques = action * self.scaling_factor * self.extrinsic_motor_strength
        return torques
