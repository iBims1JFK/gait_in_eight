from gait_in_eight.environments import observation_indices_wo_feet as obs_idx


class DefaultObservationNoise:
    def __init__(self, env,
                 joint_position_noise=0.01, joint_velocity_noise=0.5,
                 trunk_linear_velocity_noise=0.1,
                 trunk_angular_velocity_noise=0.1,
                 gravity_vector_noise=0.05):
        self.env = env
        self.joint_position_noise = joint_position_noise
        self.joint_velocity_noise = joint_velocity_noise
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
        self.trunk_linear_velocity_ids = obs_idx.TRUNK_LINEAR_VELOCITIES
        self.trunk_angular_velocity_ids = obs_idx.TRUNK_ANGULAR_VELOCITIES
        self.gravity_vector_ids = obs_idx.PROJECTED_GRAVITY

    def modify_observation(self, obs):
        obs[self.joint_position_ids] += self.env.np_rng.uniform(-self.joint_position_noise, self.joint_position_noise, self.env.nominal_joint_positions.shape[0])
        obs[self.trunk_linear_velocity_ids] += self.env.np_rng.uniform(-self.trunk_angular_velocity_noise, self.trunk_angular_velocity_noise, 3)
        obs[self.trunk_angular_velocity_ids] += self.env.np_rng.uniform(-self.trunk_angular_velocity_noise, self.trunk_angular_velocity_noise, 3)
        obs[self.joint_velocity_ids] += self.env.np_rng.uniform(-self.joint_velocity_noise, self.joint_velocity_noise, self.env.nominal_joint_positions.shape[0])
        obs[self.gravity_vector_ids] += self.env.np_rng.uniform(-self.gravity_vector_noise, self.gravity_vector_noise, 3)

        return obs
