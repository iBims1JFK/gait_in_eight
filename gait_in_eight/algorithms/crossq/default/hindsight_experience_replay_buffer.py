import numpy as np
from gait_in_eight.environments import observation_indices_wo_feet_cpg as obs_idx


class HindsightExperienceReplayBuffer():
    def __init__(self, capacity, env, nr_envs, os_shape, as_shape, rng):
        self.os_shape = os_shape
        self.as_shape = as_shape
        self.capacity = capacity // nr_envs
        self.env = env
        self.nr_envs = nr_envs
        self.rng = rng
        self.states = np.zeros((self.capacity, nr_envs) + os_shape, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, nr_envs) + os_shape, dtype=np.float32)
        self.actions = np.zeros((self.capacity, nr_envs) + as_shape, dtype=np.float32)
        self.rewards = np.zeros((self.capacity, nr_envs), dtype=np.float32)
        self.terminations = np.zeros((self.capacity, nr_envs), dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.current_episode = []
        self.mode = "avg_episode"

    def add(self, states, next_states, actions, rewards, terminations, truncated):
        self.add_step(states, next_states, actions, rewards, terminations)
        if not terminations.any() and not truncated:
            self.current_episode.append((states, next_states, actions, rewards, terminations))
        else:
            self.her()
            self.current_episode = []

    def her(self):
        if self.mode == "avg_episode":
            avg_x_vel = 0
            avg_y_vel = 0
            avg_yaw_vel = 0
            for states, _, _, _, _ in self.current_episode:
                avg_x_vel += states[0, obs_idx.TRUNK_LINEAR_VELOCITIES][0]
                avg_y_vel += states[0, obs_idx.TRUNK_LINEAR_VELOCITIES][1]
                avg_yaw_vel += states[0, obs_idx.TRUNK_ANGULAR_VELOCITIES][2]
            avg_x_vel /= len(self.current_episode)
            avg_y_vel /= len(self.current_episode)
            avg_yaw_vel /= len(self.current_episode)
            desired_local_linear_velocity_xy = np.array([avg_x_vel, avg_y_vel])
            for (states, next_states, actions, rewards, terminations) in self.current_episode:
                current_local_linear_velocity = states[0, obs_idx.TRUNK_LINEAR_VELOCITIES]
                current_local_angular_velocity = states[0, obs_idx.TRUNK_ANGULAR_VELOCITIES]
                desired_local_yaw_velocity = avg_yaw_vel

                states, rewards = self.reward(states, rewards, current_local_linear_velocity, desired_local_linear_velocity_xy, current_local_angular_velocity, desired_local_yaw_velocity)
                self.add_step(states, next_states, actions, rewards, terminations)
        else:
            raise NotImplementedError
        return states, rewards

    def add_step(self, states, next_states, actions, rewards, terminations):
        self.states[self.pos] = states
        self.next_states[self.pos] = next_states
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.terminations[self.pos] = terminations
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def reward(self, states, rewards, current_local_linear_velocity, desired_local_linear_velocity_xy, current_local_angular_velocity, desired_local_yaw_velocity):
        r = np.linalg.norm(current_local_linear_velocity[:2] - desired_local_linear_velocity_xy[:2])
        r = r / 0.5
        # r = r / (np.linalg.norm(desired_local_linear_velocity_xy[:2]) + 1)
        r = 1 - r
        r = np.clip(r, 0, 1)
        target_velocity_reward = r
        
        yaw_current_velocity = np.abs(current_local_angular_velocity[2])
        # Total reward

        reward = (
            target_velocity_reward - 0.1 * np.clip(np.abs(yaw_current_velocity - desired_local_yaw_velocity), 0, 1)
        )
        reward *= 10
        reward = max(reward, 0.0)

        states[0, obs_idx.GOAL_VELOCITIES] = np.array([*desired_local_linear_velocity_xy[:2], desired_local_yaw_velocity])
        rewards[0] = reward

        return states, rewards

    def sample(self, nr_samples):
        idx1 = self.rng.integers(self.size, size=nr_samples)
        idx2 = self.rng.integers(self.nr_envs, size=nr_samples)
        states = self.states[idx1, idx2].reshape((nr_samples,) + self.os_shape)
        next_states = self.next_states[idx1, idx2].reshape((nr_samples,) + self.os_shape)
        actions = self.actions[idx1, idx2].reshape((nr_samples,) + self.as_shape)
        rewards = self.rewards[idx1, idx2].reshape((nr_samples,))
        terminations = self.terminations[idx1, idx2].reshape((nr_samples,))
        next_observation_states = np.empty((nr_samples, self.os_shape[0] * 1))
        for i in range(nr_samples):
            next_observation_states[i] = self.states[(idx1[i] - 1 + 2):(idx1[i] + 2), idx2[i]].flatten()
        return states, next_states, actions, rewards, terminations, next_observation_states

    def replay_step(self, states, next_states, actions, rewards, terminations):
        reward_function = self.env.reward_function
        # info = {"t": self.episode_step}
        reward, info = self.reward_function.reward_and_info(None, False)
