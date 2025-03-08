import numpy as np
import time
from gait_in_eight.environments.honey_badger_ros.control_functions.inverse_kinematics import InverseKinematics
from gait_in_eight.environments.honey_badger_ros.central_pattern_generator.beat import BeatGenerator

class CPG:
    def __init__(self, env, frequency, action_space_mode):
        self.frequency = frequency
        self.i = 0
        self.update = time.perf_counter()
        self.env = env
        self.ik = InverseKinematics(env)
        self.beat = BeatGenerator(env)
        self.offset = np.array([0.0, np.pi])
        self.action_space_mode = action_space_mode
        self.reset()
    
    def step(self):
        if self.action_space_mode == "cpg_default":
            self.offset = self.beat.step(self.frequency)
            action = self.action
        elif self.action_space_mode == "cpg_frequency":
            frequency = self.action[-1]
            print(frequency)
            self.offset = self.beat.step(frequency)
            action = self.action[:-1]

        if self.action_space_mode == "cpg_residual":
            self.offset = self.beat.step(self.frequency)
            action = self.action.copy()
            action = action.reshape(-1, 6)
            joint_residuals = action[:, 3:]
            joint_residuals = joint_residuals.flatten()
            action = action[:, :3]
            action = action.flatten()

            
            # either this
            # joint_residuals += self.env.nominal_joint_positions
            # joint_target = self.ik.process_action(action, self.offset)
            # joint_target += joint_residuals
            # joint_target /= 2

            # or this
            # action space here should be a bit more restricted
            joint_target = self.ik.process_action(action, self.offset)
            joint_target += joint_residuals
        else:
            joint_target = self.ik.process_action(action, self.offset)
        return joint_target
    
    def reset(self):
        self.beat.setup()
        if self.action_space_mode == "cpg_default":
            action_shape = 12
        elif self.action_space_mode == "cpg_frequency":
            action_shape = 13
        elif self.action_space_mode == "cpg_residual":
            action_shape = 24
        else:
            raise ValueError("Invalid action space mode")
        self.action = np.zeros(action_shape)
    
    def get_action(self):
        return self.action
    
    def set_action(self, action):
        self.action = action
    
    def get_offset(self):
        return self.offset  