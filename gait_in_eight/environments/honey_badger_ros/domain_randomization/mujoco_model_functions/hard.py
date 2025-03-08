class HardDomainMuJoCoModel:
    def __init__(self, env,
                 friction_tangential_min=0.001, friction_tangential_max=2.0,
                 friction_torsional_ground_min=0.00001, friction_torsional_ground_max=0.01,
                 friction_torsional_feet_min=0.00001, friction_torsional_feet_max=0.04,
                 friction_rolling_ground_min=0.00001, friction_rolling_ground_max=0.0002,
                 friction_rolling_feet_min=0.00001, friction_rolling_feet_max=0.02,
                 damping_min=30, damping_max=130,
                 stiffness_min=500, stiffness_max=1500,
                 gravity_min=8.81, gravity_max=10.81,
                 add_trunk_mass_min=-2.0, add_trunk_mass_max=2.0,
                 add_com_displacement_min=-0.01, add_com_displacement_max=0.01,
                 foot_size_min=0.018, foot_size_max=0.022,
                 joint_damping_min=0.0, joint_damping_max=2.0,
                 joint_armature_min=0.008, joint_armature_max=0.05,
                 joint_stiffness_min=0.0, joint_stiffness_max=2.0,
                 joint_friction_loss_min=0.0, joint_friction_loss_max=1.0,
        ):
        self.env = env
        self.friction_tangential_min = friction_tangential_min
        self.friction_tangential_max = friction_tangential_max
        self.friction_torsional_ground_min = friction_torsional_ground_min
        self.friction_torsional_ground_max = friction_torsional_ground_max
        self.friction_torsional_feet_min = friction_torsional_feet_min
        self.friction_torsional_feet_max = friction_torsional_feet_max
        self.friction_rolling_ground_min = friction_rolling_ground_min
        self.friction_rolling_ground_max = friction_rolling_ground_max
        self.friction_rolling_feet_min = friction_rolling_feet_min
        self.friction_rolling_feet_max = friction_rolling_feet_max
        self.damping_min = damping_min
        self.damping_max = damping_max
        self.stiffness_min = stiffness_min
        self.stiffness_max = stiffness_max
        self.gravity_min = gravity_min
        self.gravity_max = gravity_max
        self.add_trunk_mass_min = add_trunk_mass_min
        self.add_trunk_mass_max = add_trunk_mass_max
        self.add_com_displacement_min = add_com_displacement_min
        self.add_com_displacement_max = add_com_displacement_max
        self.foot_size_min = foot_size_min
        self.foot_size_max = foot_size_max
        self.joint_damping_min = joint_damping_min
        self.joint_damping_max = joint_damping_max
        self.joint_armature_min = joint_armature_min
        self.joint_armature_max = joint_armature_max
        self.joint_stiffness_min = joint_stiffness_min
        self.joint_stiffness_max = joint_stiffness_max
        self.joint_friction_loss_min = joint_friction_loss_min
        self.joint_friction_loss_max = joint_friction_loss_max
    
    def init(self):
        self.default_trunk_mass = self.env.model.body_mass[1]
        self.default_trunk_inertia = self.env.model.body_inertia[1].copy()
        self.default_trunk_com = self.env.model.body_ipos[1].copy()
        self.default_joint_damping = self.env.model.dof_damping[6:].copy()
        self.default_joint_armature = self.env.model.dof_armature[6:].copy()
        self.default_joint_stiffness = self.env.model.jnt_stiffness[1:].copy()
        self.default_joint_frictionloss = self.env.model.dof_frictionloss[6:].copy()

    def sample(self):
        interpolation = self.env.np_rng.uniform(0, 1)
        self.sampled_friction_tangential = self.friction_tangential_min + (self.friction_tangential_max - self.friction_tangential_min) * interpolation
        self.sampled_friction_torsional_ground = self.friction_torsional_ground_min + (self.friction_torsional_ground_max - self.friction_torsional_ground_min) * interpolation
        self.sampled_friction_torsional_feet = self.friction_torsional_feet_min + (self.friction_torsional_feet_max - self.friction_torsional_feet_min) * interpolation
        self.sampled_friction_rolling_ground = self.friction_rolling_ground_min + (self.friction_rolling_ground_max - self.friction_rolling_ground_min) * interpolation
        self.sampled_friction_rolling_feet = self.friction_rolling_feet_min + (self.friction_rolling_feet_max - self.friction_rolling_feet_min) * interpolation
        self.env.model.geom_friction[0] = [self.sampled_friction_tangential, self.sampled_friction_torsional_ground, self.sampled_friction_rolling_ground]
        self.env.model.geom_friction[[10, 18, 26, 34]] = [self.sampled_friction_tangential, self.sampled_friction_torsional_feet, self.sampled_friction_rolling_feet]

        interpolation = self.env.np_rng.uniform(0, 1)
        self.sampled_damping = self.damping_min + (self.damping_max - self.damping_min) * interpolation
        self.sampled_stiffness = self.stiffness_min + (self.stiffness_max - self.stiffness_min) * interpolation
        self.env.model.geom_solref[:, 0] = -self.sampled_stiffness
        self.env.model.geom_solref[:, 1] = -self.sampled_damping

        self.sampled_gravity = self.env.np_rng.uniform(self.gravity_min, self.gravity_max)
        self.env.model.opt.gravity[2] = -self.sampled_gravity

        self.env.model.body_mass[1] = self.default_trunk_mass + self.env.np_rng.uniform(self.add_trunk_mass_min, self.add_trunk_mass_max)
        self.env.model.body_inertia[1] = self.default_trunk_inertia * (self.env.model.body_mass[1] / self.default_trunk_mass)

        self.env.model.body_ipos[1] = self.default_trunk_com + self.env.np_rng.uniform(self.add_com_displacement_min, self.add_com_displacement_max)

        self.env.model.geom_size[[10, 18, 26, 34],0] = self.env.np_rng.uniform(self.foot_size_min, self.foot_size_max)

        if self.env.np_rng.uniform(0, 1) > 0.5:
            self.env.model.dof_damping[6:] = self.env.np_rng.uniform(self.joint_damping_min, self.joint_damping_max, size=self.default_joint_damping.shape)
            self.env.model.dof_armature[6:] = self.env.np_rng.uniform(self.joint_armature_min, self.joint_armature_max, size=self.default_joint_armature.shape)
            self.env.model.jnt_stiffness[1:] = self.env.np_rng.uniform(self.joint_stiffness_min, self.joint_stiffness_max, size=self.default_joint_stiffness.shape)
            self.env.model.dof_frictionloss[6:] = self.env.np_rng.uniform(self.joint_friction_loss_min, self.joint_friction_loss_max, size=self.default_joint_frictionloss.shape)
        else:
            self.env.model.dof_damping[6:] = self.default_joint_damping
            self.env.model.dof_armature[6:] = self.default_joint_armature
            self.env.model.jnt_stiffness[1:] = self.default_joint_stiffness
            self.env.model.dof_frictionloss[6:] = self.default_joint_frictionloss
