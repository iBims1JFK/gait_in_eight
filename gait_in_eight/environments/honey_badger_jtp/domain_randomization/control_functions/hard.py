class HardDomainControl:
    def __init__(self, env,
                 motor_strength_min=0.5, motor_strength_max=1.5,
                 p_gain_factor_min=0.5, p_gain_factor_max=1.5,
                 d_gain_factor_min=0.5, d_gain_factor_max=1.5,
                 asymmetric_factor_min=0.95, asymmetric_factor_max=1.05,
                 position_offset_min=-0.05, position_offset_max=0.05,
                 ):
        self.env = env
        self.motor_strength_min = motor_strength_min
        self.motor_strength_max = motor_strength_max
        self.p_gain_factor_min = p_gain_factor_min
        self.p_gain_factor_max = p_gain_factor_max
        self.d_gain_factor_min = d_gain_factor_min
        self.d_gain_factor_max = d_gain_factor_max
        self.asymmetric_factor_min = asymmetric_factor_min
        self.asymmetric_factor_max = asymmetric_factor_max
        self.position_offset_min = position_offset_min
        self.position_offset_max = position_offset_max

    def sample(self):
        self.sampled_motor_strength = self.env.np_rng.uniform(self.motor_strength_min, self.motor_strength_max)
        self.sampled_asymmetric_factor_m = self.env.np_rng.uniform(self.asymmetric_factor_min, self.asymmetric_factor_max, self.env.model.nu)
        self.env.control_function.extrinsic_motor_strength = self.sampled_motor_strength * self.sampled_asymmetric_factor_m

        self.sampled_p_gain_factor = self.env.np_rng.uniform(self.p_gain_factor_min, self.p_gain_factor_max)
        self.sampled_asymmetric_factor_p = self.env.np_rng.uniform(self.asymmetric_factor_min, self.asymmetric_factor_max, self.env.model.nu)
        self.env.control_function.extrinsic_p_gain_factor = self.sampled_p_gain_factor * self.sampled_asymmetric_factor_p

        self.sampled_d_gain_factor = self.env.np_rng.uniform(self.d_gain_factor_min, self.d_gain_factor_max)
        self.sampled_asymmetric_factor_d = self.env.np_rng.uniform(self.asymmetric_factor_min, self.asymmetric_factor_max, self.env.model.nu)
        self.env.control_function.extrinsic_d_gain_factor = self.sampled_d_gain_factor * self.sampled_asymmetric_factor_d

        self.sampled_position_offset = self.env.np_rng.uniform(self.position_offset_min, self.position_offset_max, self.env.model.nu)
        self.env.control_function.extrinsic_position_offset = self.sampled_position_offset
