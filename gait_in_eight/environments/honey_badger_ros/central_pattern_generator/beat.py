import numpy as np
import time
class BeatGenerator:
    def __init__(self, env):
        self.env = env
        self.target_frequency = 1
        self.control_frequency = 40
        self.i = 0
        self.update = time.perf_counter()

    
    def setup(self):
        self.i = 0
        self.update = time.perf_counter()

    def step(self, frequency=2.00):
        offset = np.zeros(2)
        t1 = (self.i * 2 * np.pi) % (2 * np.pi)
        t2 = (self.i * 2 * np.pi + np.pi) % (2 * np.pi)
        if t1 <= np.pi / 2:
            offset[0] += self.swing_up(t1)
        elif t1 <= np.pi:
            offset[0] += self.swing_down(t1)
        
        if t2 <= np.pi / 2:
            offset[1] += self.swing_up(t2)
        elif t2 <= np.pi:
            offset[1] += self.swing_down(t2)
        
        current_time = time.perf_counter()
        self.i += (current_time - self.update) * frequency
        self.update = current_time
        return offset

    def swing_up(self, phi_l):
        t_l = 2 / np.pi * phi_l
        return 0.2 * (-2 * t_l**3 + 3 * t_l**2)

    def swing_down(self, phi_l):
        t_l = 2 / np.pi * phi_l - 1
        return 0.2 * (2 * t_l**3 - 3 * t_l**2 + 1)
    
