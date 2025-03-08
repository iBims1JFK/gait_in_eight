import numpy as np


class PDControl:
    def __init__(self, env, control_frequency_hz=50, p_gain=25.0, d_gain=1.0, scaling_factor=1.0):
        self.env = env
        self.control_frequency_hz = control_frequency_hz
        self.p_gain = p_gain
        self.d_gain = d_gain
        self.scaling_factor = scaling_factor
        self.extrinsic_p_gain_factor = np.ones(12)
        self.extrinsic_d_gain_factor = np.ones(12)
        self.motor_strength_factor = np.ones(12)
        self.extrinsic_position_offset = np.zeros(12)

    def process_action(self, action):
        target_joint_positions = self.env.nominal_joint_positions + action * self.scaling_factor
        torques = self.p_gain * self.extrinsic_p_gain_factor * (target_joint_positions - self.env.data.qpos[7:] + self.extrinsic_position_offset) \
                  - self.d_gain * self.extrinsic_d_gain_factor * self.env.data.qvel[6:]
        torques = torques * self.motor_strength_factor
        return torques
