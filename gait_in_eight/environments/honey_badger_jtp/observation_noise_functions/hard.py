from gait_in_eight.environments import observation_indices_wo_feet as obs_idx
import numpy as np

class HardObservationNoise:
    def __init__(self, env,
                 joint_position_noise=0.01, joint_velocity_noise=1.5,
                 trunk_linear_acceleration_noise=0.05, trunk_angular_acceleration_noise=0.05,
                 trunk_linear_velocity_noise=0.2,
                 trunk_angular_velocity_noise=0.2,
                 gravity_vector_noise=0.1):
        self.env = env
        self.joint_position_noise = joint_position_noise
        self.joint_velocity_noise = joint_velocity_noise
        self.trunk_linear_acceleration_noise = trunk_linear_acceleration_noise
        self.trunk_angular_acceleration_noise = trunk_angular_acceleration_noise
        self.trunk_linear_velocity_noise = trunk_linear_velocity_noise
        self.trunk_angular_velocity_noise = trunk_angular_velocity_noise
        self.gravity_vector_noise = gravity_vector_noise

    def init(self):
        self.joint_position_ids = [
            obs_idx.QUADRUPED_FRONT_LEFT_HIP[0], obs_idx.QUADRUPED_FRONT_LEFT_THIGH[0], obs_idx.QUADRUPED_FRONT_LEFT_CALF[0],
            obs_idx.QUADRUPED_FRONT_RIGHT_HIP[0], obs_idx.QUADRUPED_FRONT_RIGHT_THIGH[0], obs_idx.QUADRUPED_FRONT_RIGHT_CALF[0],
            obs_idx.QUADRUPED_BACK_LEFT_HIP[0], obs_idx.QUADRUPED_BACK_LEFT_THIGH[0], obs_idx.QUADRUPED_BACK_LEFT_CALF[0],
            obs_idx.QUADRUPED_BACK_RIGHT_HIP[0], obs_idx.QUADRUPED_BACK_RIGHT_THIGH[0], obs_idx.QUADRUPED_BACK_RIGHT_CALF[0],
        ]
        self.joint_velocity_ids = [
            obs_idx.QUADRUPED_FRONT_LEFT_HIP[1], obs_idx.QUADRUPED_FRONT_LEFT_THIGH[1], obs_idx.QUADRUPED_FRONT_LEFT_CALF[1],
            obs_idx.QUADRUPED_FRONT_RIGHT_HIP[1], obs_idx.QUADRUPED_FRONT_RIGHT_THIGH[1], obs_idx.QUADRUPED_FRONT_RIGHT_CALF[1],
            obs_idx.QUADRUPED_BACK_LEFT_HIP[1], obs_idx.QUADRUPED_BACK_LEFT_THIGH[1], obs_idx.QUADRUPED_BACK_LEFT_CALF[1],
            obs_idx.QUADRUPED_BACK_RIGHT_HIP[1], obs_idx.QUADRUPED_BACK_RIGHT_THIGH[1], obs_idx.QUADRUPED_BACK_RIGHT_CALF[1],
        ]
        self.trunk_acceleration_ids = obs_idx.TRUNK_LINEAR_ACCELERATIONS
        self.trunk_angular_acceleration_ids = obs_idx.TRUNK_ANGULAR_ACCELERATIONS
        self.trunk_linear_velocity_ids = obs_idx.TRUNK_LINEAR_VELOCITIES
        self.trunk_angular_velocity_ids = obs_idx.TRUNK_ANGULAR_VELOCITIES
        self.gravity_vector_ids = obs_idx.PROJECTED_GRAVITY
        self.target_velocity_bias = self.env.np_rng.uniform(-0.0003, 0.0003, 3)
        self.velocity_bias = np.zeros(3)

    def modify_observation(self, obs):
        if self.env.episode_step == 0:
            self.target_velocity_bias = self.env.np_rng.uniform(-0.0003, 0.0003, 3)
            self.velocity_bias = self.env.np_rng.normal(self.target_velocity_bias, 0.00134, 3)
        else:
            self.velocity_bias += self.env.np_rng.normal(self.target_velocity_bias, 0.00134, 3)
        
        obs[self.joint_position_ids] += self.env.np_rng.uniform(-self.joint_position_noise, self.joint_position_noise, self.env.nominal_joint_positions.shape[0])
        obs[self.trunk_acceleration_ids] += self.env.np_rng.uniform(-self.trunk_linear_acceleration_noise, self.trunk_linear_acceleration_noise, 3)
        obs[self.trunk_angular_acceleration_ids] += self.env.np_rng.uniform(-self.trunk_angular_acceleration_noise, self.trunk_angular_acceleration_noise, 3)
        obs[self.trunk_linear_velocity_ids] += self.env.np_rng.normal(0, 0.00134, 3) + self.velocity_bias
        obs[self.trunk_angular_velocity_ids] += self.env.np_rng.uniform(-self.trunk_angular_velocity_noise, self.trunk_angular_velocity_noise, 3)
        obs[self.joint_velocity_ids] += self.env.np_rng.uniform(-self.joint_velocity_noise, self.joint_velocity_noise, self.env.nominal_joint_positions.shape[0])
        obs[self.gravity_vector_ids] += self.env.np_rng.uniform(-self.gravity_vector_noise, self.gravity_vector_noise, 3)

        return obs
