import numpy as np


class ReplayBuffer():
    def __init__(self, capacity, nr_envs, os_shape, as_shape, rng, history_length=1):
        self.os_shape = os_shape
        self.as_shape = as_shape
        self.capacity = capacity // nr_envs
        self.nr_envs = nr_envs
        self.rng = rng
        self.states = np.zeros((self.capacity, nr_envs) + os_shape, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, nr_envs) + os_shape, dtype=np.float32)
        self.actions = np.zeros((self.capacity, nr_envs) + as_shape, dtype=np.float32)
        self.rewards = np.zeros((self.capacity, nr_envs), dtype=np.float32)
        self.terminations = np.zeros((self.capacity, nr_envs), dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.history_length = history_length
    

    def add(self, states, next_states, actions, rewards, terminations):
        self.states[self.pos] = states
        self.next_states[self.pos] = next_states
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.terminations[self.pos] = terminations
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    

    def sample(self, nr_samples):
        idx1 = self.rng.integers(low=self.history_length, high=self.size - 1, size=nr_samples)
        idx2 = self.rng.integers(self.nr_envs, size=nr_samples)
        states = np.empty((nr_samples, self.os_shape[0] * self.history_length))
        for i in range(nr_samples):
            states[i] = self.states[(idx1[i] - self.history_length + 1):(idx1[i] + 1), idx2[i]].flatten()
        next_observation_states = np.empty((nr_samples, self.os_shape[0] * self.history_length))
        for i in range(nr_samples):
            next_observation_states[i] = self.states[(idx1[i] - self.history_length + 2):(idx1[i] + 2), idx2[i]].flatten()
        next_states = self.next_states[idx1, idx2].reshape((nr_samples,) + self.os_shape)
        actions = self.actions[idx1, idx2].reshape((nr_samples,) + self.as_shape)
        rewards = self.rewards[idx1, idx2].reshape((nr_samples,))
        terminations = self.terminations[idx1, idx2].reshape((nr_samples,))
        return states, next_states, actions, rewards, terminations, next_observation_states
