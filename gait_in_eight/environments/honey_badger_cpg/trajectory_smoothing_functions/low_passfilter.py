import numpy as np
import scipy.signal as signal

class LowPassFilter:
    def __init__(self, env):
        self.env = env
        frequency = 40
        nyquist = 0.5 * frequency
        low = 4 / nyquist
        self.b, self.a = signal.butter(2, low, 'low', analog=False)

    def init(self):
        self.action_space = self.env.model.nu
        self.z = [np.zeros(2) for _ in range(self.action_space)]

    def step(self, action):
        filter_action = action.copy()
        for i in range(self.action_space):
            filter_action[i], self.z[i] = signal.lfilter(self.b, self.a, [action[i]], zi=self.z[i])
        return filter_action