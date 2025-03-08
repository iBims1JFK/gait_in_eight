import numpy as np


class MaxXReward:
    def __init__(self, env,
                 tracking_xy_velocity_command_coeff_start=4.0, tracking_xy_velocity_command_coeff_end=2.0,
                 tracking_yaw_velocity_command_coeff_start=0.1, tracking_yaw_velocity_command_coeff_end=1.0,
                 curriculum_start_steps=0e6, curriculum_end_steps=10e6,
                 xy_tracking_temperature=0.25, yaw_tracking_temperature=0.25,
                 z_velocity_coeff=2e0, pitch_roll_vel_coeff=5e-2, pitch_roll_pos_coeff=2e-1, joint_position_limit_coeff=1e1, soft_joint_position_limit=0.9,
                 joint_velocity_coeff=0.0, joint_acceleration_coeff=2.5e-7, joint_torque_coeff=2e-4, action_rate_coeff=1e-2,
                 collision_coeff=1e0, base_height_coeff=3e1, nominal_trunk_z=0.316, air_time_coeff=1e-1, air_time_max=0.5, symmetry_air_coeff=0.5):
        self.env = env
        self.tracking_xy_velocity_command_coeff_start = tracking_xy_velocity_command_coeff_start * self.env.dt
        self.tracking_xy_velocity_command_coeff_end = tracking_xy_velocity_command_coeff_end * self.env.dt
        self.tracking_yaw_velocity_command_coeff_start = tracking_yaw_velocity_command_coeff_start * self.env.dt
        self.tracking_yaw_velocity_command_coeff_end = tracking_yaw_velocity_command_coeff_end * self.env.dt
        self.curriculum_start_steps = curriculum_start_steps
        self.curriculum_end_steps = curriculum_end_steps
        self.xy_tracking_temperature = xy_tracking_temperature
        self.yaw_tracking_temperature = yaw_tracking_temperature
        self.z_velocity_coeff = z_velocity_coeff * self.env.dt
        self.pitch_roll_vel_coeff = pitch_roll_vel_coeff * self.env.dt
        self.pitch_roll_pos_coeff = pitch_roll_pos_coeff * self.env.dt
        self.joint_position_limit_coeff = joint_position_limit_coeff * self.env.dt
        self.soft_joint_position_limit = soft_joint_position_limit
        self.joint_velocity_coeff = joint_velocity_coeff * self.env.dt
        self.joint_acceleration_coeff = joint_acceleration_coeff * self.env.dt
        self.joint_torque_coeff = joint_torque_coeff * self.env.dt
        self.action_rate_coeff = action_rate_coeff * self.env.dt
        self.collision_coeff = collision_coeff * self.env.dt
        self.base_height_coeff = base_height_coeff * self.env.dt
        self.nominal_trunk_z = nominal_trunk_z
        self.air_time_coeff = air_time_coeff * self.env.dt
        self.air_time_max = air_time_max
        self.symmetry_air_coeff = symmetry_air_coeff * self.env.dt
        self.max_vel_reached = 0.0

        self.time_since_last_touchdown_fr = 0
        self.time_since_last_touchdown_fl = 0
        self.time_since_last_touchdown_rr = 0
        self.time_since_last_touchdown_rl = 0
        self._prev_joint_vel = None

    def init(self):
        self.joint_limits = self.env.model.jnt_range[1:].copy()
        joint_limits_midpoint = (self.joint_limits[:, 0] + self.joint_limits[:, 1]) / 2
        joint_limits_range = self.joint_limits[:, 1] - self.joint_limits[:, 0]
        self.joint_limits[:, 0] = joint_limits_midpoint - joint_limits_range / 2 * self.soft_joint_position_limit
        self.joint_limits[:, 1] = joint_limits_midpoint + joint_limits_range / 2 * self.soft_joint_position_limit

    def setup(self):
        self.time_since_last_touchdown_fr = 0
        self.time_since_last_touchdown_fl = 0
        self.time_since_last_touchdown_rr = 0
        self.time_since_last_touchdown_rl = 0
        self.prev_joint_vel = np.zeros(self.env.model.nu)
        self.sum_tracking_performance_percentage = 0.0

    def step(self, action):
        self.time_since_last_touchdown_fr = 0 if self.env.check_collision("floor", "FR_foot") else self.time_since_last_touchdown_fr + self.env.dt
        self.time_since_last_touchdown_fl = 0 if self.env.check_collision("floor", "FL_foot") else self.time_since_last_touchdown_fl + self.env.dt
        self.time_since_last_touchdown_rr = 0 if self.env.check_collision("floor", "RR_foot") else self.time_since_last_touchdown_rr + self.env.dt
        self.time_since_last_touchdown_rl = 0 if self.env.check_collision("floor", "RL_foot") else self.time_since_last_touchdown_rl + self.env.dt
        self.prev_joint_vel = np.array(self.env.data.qvel[6:])

    def reward_and_info(self, info, done):
        total_timesteps = self.env.total_timesteps * self.env.total_nr_envs

        # Tracking velocity command reward
        current_global_linear_velocity = self.env.data.qvel[:3]
        current_local_linear_velocity = self.env.orientation_quat_inv.apply(current_global_linear_velocity)
        desired_local_linear_velocity_xy = np.array([self.env.goal_x_velocity, self.env.goal_y_velocity])
        xy_velocity_difference_norm = np.sum(np.square(desired_local_linear_velocity_xy - current_local_linear_velocity[:2]))
        tracking_xy_velocity_command_coeff = self.tracking_xy_velocity_command_coeff_start + \
                                                (self.tracking_xy_velocity_command_coeff_end - self.tracking_xy_velocity_command_coeff_start) * \
                                                min(total_timesteps / self.curriculum_end_steps, 1.0)
        tracking_xy_velocity_command_reward = (tracking_xy_velocity_command_coeff *
                                               np.exp(-xy_velocity_difference_norm / self.xy_tracking_temperature))

        # Tracking angular velocity command reward
        current_local_angular_velocity = self.env.data.qvel[3:6]
        desired_local_yaw_velocity = self.env.goal_yaw_velocity
        yaw_velocity_difference_norm = np.sum(np.square(current_local_angular_velocity[2] - desired_local_yaw_velocity))
        tracking_yaw_velocity_command_coeff = self.tracking_yaw_velocity_command_coeff_start + \
                                                (self.tracking_yaw_velocity_command_coeff_end - self.tracking_yaw_velocity_command_coeff_start) * \
                                                min(total_timesteps / self.curriculum_end_steps, 1.0)
        tracking_yaw_velocity_command_reward = (tracking_yaw_velocity_command_coeff *
                                                np.exp(-yaw_velocity_difference_norm / self.yaw_tracking_temperature))

        # Linear velocity reward
        z_velocity_squared = current_local_linear_velocity[2] ** 2
        linear_velocity_reward = self.z_velocity_coeff * -z_velocity_squared

        # Angular velocity reward
        angular_velocity_norm = np.sum(np.square(current_local_angular_velocity[:2]))
        angular_velocity_reward = self.pitch_roll_vel_coeff * -angular_velocity_norm

        # Angular position reward
        orientation_euler = self.env.orientation_quat.as_euler("xyz")
        pitch_roll_position_norm = np.sum(np.square(orientation_euler[:2]))
        angular_position_reward = self.pitch_roll_pos_coeff * -pitch_roll_position_norm

        # Joint position limit reward
        joint_positions = np.array(self.env.data.qpos[7:])
        lower_limit_penalty = -np.minimum(joint_positions - self.joint_limits[:, 0], 0.0).sum()
        upper_limit_penalty = np.maximum(joint_positions - self.joint_limits[:, 1], 0.0).sum()
        joint_position_limit_reward = self.joint_position_limit_coeff * -(lower_limit_penalty + upper_limit_penalty)

        # Joint velocity reward
        velocity_norm = np.sum(np.square(np.array(self.env.data.qvel[6:])))
        velocity_reward = self.joint_velocity_coeff * -velocity_norm

        # Joint acceleration reward
        acceleration_norm = np.sum(np.square(self.prev_joint_vel - np.array(self.env.data.qvel[6:]) / self.env.dt))
        acceleration_reward = self.joint_acceleration_coeff * -acceleration_norm

        # Joint torque reward
        torque_norm = np.sum(np.square(np.array(self.env.data.qfrc_actuator[6:])))
        torque_reward = self.joint_torque_coeff * -torque_norm

        # Action rate reward
        action_rate_norm = np.sum(np.square(self.env.current_action - self.env.last_action))
        action_rate_reward = self.action_rate_coeff * -action_rate_norm

        # Collision reward
        collisions = self.env.check_any_collision_for_all([
            "FR_calf", "FL_calf", "RR_calf", "RL_calf"
        ])
        trunk_collision = 1 if self.env.check_any_collision(["trunk_1",]) else 0
        nr_collisions = sum(collisions.values()) + trunk_collision
        collision_reward = self.collision_coeff * -nr_collisions

        # Walking height
        trunk_z = self.env.data.qpos[2] - self.env.terrain_function.center_height
        height_difference_squared = (trunk_z - self.nominal_trunk_z) ** 2
        base_height_reward = self.base_height_coeff * -height_difference_squared

        # Air time reward
        air_time_reward = 0.0
        foot_fr_on_ground = self.env.check_collision("floor", "FR_foot")
        foot_fl_on_ground = self.env.check_collision("floor", "FL_foot")
        foot_rr_on_ground = self.env.check_collision("floor", "RR_foot")
        foot_rl_on_ground = self.env.check_collision("floor", "RL_foot")
        if foot_fr_on_ground:
            air_time_reward += self.time_since_last_touchdown_fr - self.air_time_max
        if foot_fl_on_ground:
            air_time_reward += self.time_since_last_touchdown_fl - self.air_time_max
        if foot_rr_on_ground:
            air_time_reward += self.time_since_last_touchdown_rr - self.air_time_max
        if foot_rl_on_ground:
            air_time_reward += self.time_since_last_touchdown_rl - self.air_time_max
        air_time_reward = self.air_time_coeff * air_time_reward

        # Symmetry reward
        symmetry_air_violations = 0.0
        if not foot_fr_on_ground and not foot_fl_on_ground:
            symmetry_air_violations += 1
        if not foot_rr_on_ground and not foot_rl_on_ground:
            symmetry_air_violations += 1
        symmetry_air_reward = self.symmetry_air_coeff * -symmetry_air_violations

        r = current_local_linear_velocity[0] - np.abs(current_local_linear_velocity[1])
        r = r / 0.5
        target_velocity_reward = r
        
        yaw_penalty = -0.1 * np.abs(current_local_angular_velocity[2])

        local_acceleration = self.env.data.qacc[:3]
        global_acceleration = self.env.orientation_quat.apply(local_acceleration)
        z_acceleration = global_acceleration[2]
        z_acceleration_penalty = -np.abs(z_acceleration) * 0.1
        
        orientation_euler = self.env.orientation_quat.as_euler("xyz")
        pitch_roll_position_norm = np.sum(np.square(orientation_euler[:2]))
        pitch_roll_position_penalty = -pitch_roll_position_norm * 10
        
        # Total reward
        reward = (
            target_velocity_reward + yaw_penalty + pitch_roll_position_penalty
        )
        reward *= 10
        # reward = max(reward, 0.0)

        if self.max_vel_reached < current_local_linear_velocity[0]:
            self.max_vel_reached = current_local_linear_velocity[0]

        # More logging metrics
        power = np.sum(abs(self.env.current_torques) * abs(self.env.data.qvel[6:]))
        mass_of_robot = np.sum(self.env.model.body_mass)
        gravity = -self.env.model.opt.gravity[2]
        velocity = np.linalg.norm(current_local_linear_velocity)
        cost_of_transport = power / (mass_of_robot * gravity * velocity)
        froude_number = velocity ** 2 / (gravity * trunk_z)
        current_global_velocities = np.array([current_local_linear_velocity[0], current_local_linear_velocity[1], current_local_angular_velocity[2]])
        desired_global_velocities = np.array([desired_local_linear_velocity_xy[0], desired_local_linear_velocity_xy[1], desired_local_yaw_velocity])
        tracking_performance_percentage = max(np.mean(1 - (np.abs(current_global_velocities - desired_global_velocities) / np.abs(desired_global_velocities))), 0.0)
        self.sum_tracking_performance_percentage += tracking_performance_percentage
        if done:
            episode_tracking_performance_percentage = self.sum_tracking_performance_percentage / self.env.horizon

        # Info
        info[f"reward/target_velocity"] = reward
        info[f"reward/max_vel/target_velocity_reward"] = target_velocity_reward
        info[f"reward/max_vel/yaw_penalty"] = yaw_penalty
        info[f"reward/max_vel/pitch_roll_position_penalty"] = pitch_roll_position_penalty
        info[f"reward/max_vel/z_acceleration_penalty"] = z_acceleration_penalty
        info[f"reward/max_vel/max_vel_reached"] = self.max_vel_reached
        info[f"reward/track_xy_vel_cmd"] = tracking_xy_velocity_command_reward
        info[f"reward/track_yaw_vel_cmd"] = tracking_yaw_velocity_command_reward
        info[f"reward/linear_velocity"] = linear_velocity_reward
        info[f"reward/angular_velocity"] = angular_velocity_reward
        info[f"reward/angular_position"] = angular_position_reward
        info[f"reward/joint_position_limit"] = joint_position_limit_reward
        info[f"reward/torque"] = torque_reward
        info[f"reward/acceleration"] = acceleration_reward
        info[f"reward/velocity"] = velocity_reward
        info[f"reward/action_rate"] = action_rate_reward
        info[f"reward/collision"] = collision_reward
        info[f"reward/base_height"] = base_height_reward
        info[f"reward/air_time"] = air_time_reward
        info[f"reward/symmetry_air"] = symmetry_air_reward
        info["env_info/target_x_vel"] = desired_local_linear_velocity_xy[0]
        info["env_info/target_y_vel"] = desired_local_linear_velocity_xy[1]
        info["env_info/target_yaw_vel"] = desired_local_yaw_velocity
        info["env_info/current_x_vel"] = current_local_linear_velocity[0]
        info["env_info/current_y_vel"] = current_local_linear_velocity[1]
        info["env_info/current_yaw_vel"] = current_local_angular_velocity[2]
        info[f"env_info/track_perf_perc"] = tracking_performance_percentage
        if done:
            info[f"env_info/eps_track_perf_perc"] = episode_tracking_performance_percentage
        info["env_info/symmetry_violations"] = symmetry_air_violations
        info["env_info/walk_height"] = trunk_z
        info["env_info/xy_vel_diff_norm"] = xy_velocity_difference_norm
        info["env_info/yaw_vel_diff_norm"] = yaw_velocity_difference_norm
        info["env_info/torque_norm"] = torque_norm
        info["env_info/acceleration_norm"] = acceleration_norm
        info["env_info/velocity_norm"] = velocity_norm
        info["env_info/action_rate_norm"] = action_rate_norm
        info["env_info/power"] = power
        info["env_info/cost_of_transport"] = cost_of_transport
        info["env_info/froude_number"] = froude_number

        return reward, info
