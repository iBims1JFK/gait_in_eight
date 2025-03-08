import numpy as np

from gait_in_eight.environments import observation_indices as obs_idx


class HardObservationNoise:
    def __init__(self, env,
                 joint_position_noise=0.01, joint_velocity_noise=1.5,
                 trunk_angular_velocity_noise=0.2,
                 ground_contact_noise_chance=0.05,
                 contact_time_noise_chance=0.05, contact_time_noise_factor=1.0,
                 gravity_vector_noise=0.05):
        self.env = env
        self.joint_position_noise = joint_position_noise
        self.joint_velocity_noise = joint_velocity_noise
        self.trunk_angular_velocity_noise = trunk_angular_velocity_noise
        self.ground_contact_noise_chance = ground_contact_noise_chance
        self.contact_time_noise_chance = contact_time_noise_chance
        self.contact_time_noise_factor = contact_time_noise_factor
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
        self.foot_contact_ids = [
            obs_idx.QUADRUPED_FRONT_LEFT_FOOT[0], obs_idx.QUADRUPED_FRONT_RIGHT_FOOT[0],
            obs_idx.QUADRUPED_BACK_LEFT_FOOT[0], obs_idx.QUADRUPED_BACK_RIGHT_FOOT[0],
        ]
        self.foot_time_since_last_touchdown_ids = [
            obs_idx.QUADRUPED_FRONT_LEFT_FOOT[1], obs_idx.QUADRUPED_FRONT_RIGHT_FOOT[1],
            obs_idx.QUADRUPED_BACK_LEFT_FOOT[1], obs_idx.QUADRUPED_BACK_RIGHT_FOOT[1],
        ]
        self.trunk_angular_velocity_ids = obs_idx.TRUNK_ANGULAR_VELOCITIES
        self.gravity_vector_ids = obs_idx.PROJECTED_GRAVITY

    def modify_observation(self, obs):
        obs[self.joint_position_ids] += self.env.np_rng.uniform(-self.joint_position_noise, self.joint_position_noise, self.env.nominal_joint_positions.shape[0])
        obs[self.trunk_angular_velocity_ids] += self.env.np_rng.uniform(-self.trunk_angular_velocity_noise, self.trunk_angular_velocity_noise, 3)
        obs[self.joint_velocity_ids] += self.env.np_rng.uniform(-self.joint_velocity_noise, self.joint_velocity_noise, self.env.nominal_joint_positions.shape[0])
        obs[self.gravity_vector_ids] += self.env.np_rng.uniform(-self.gravity_vector_noise, self.gravity_vector_noise, 3)
        obs[self.foot_contact_ids] = 1 - obs[self.foot_contact_ids] * (self.env.np_rng.uniform(0, 1, size=len(self.foot_contact_ids)) < self.ground_contact_noise_chance)
        obs[self.foot_time_since_last_touchdown_ids] += self.env.np_rng.uniform(-self.contact_time_noise_factor * self.env.dt, self.contact_time_noise_factor * self.env.dt, size=len(self.foot_time_since_last_touchdown_ids)) * (self.env.np_rng.uniform(0, 1, size=len(self.foot_time_since_last_touchdown_ids)) < self.contact_time_noise_chance)

        return obs
