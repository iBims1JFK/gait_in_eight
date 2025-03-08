from ml_collections import config_dict


def get_config(environment_name):
    config = config_dict.ConfigDict()

    config.name = environment_name

    config.nr_envs = 1

    config.seed = 1
    config.async_skip_percentage = 0.0
    config.cycle_cpu_affinity = False
    config.render = False
    config.mode = "train"                                                           # "train", "test"
    config.control_type = "pd"                                                      # "pd", "torque"
    config.control_frequency_hz = 40                                                # only used for pd_clip
    config.command_type = "forward"                                                 # "random"
    config.target_velocity = 1.0                                                    # only used for straight_ahead
    config.command_sampling_type = "only_setup"                                     # "step_probability", "none"
    config.initial_state_type = "default"                                           # "default", "random"
    config.reward_type = "track_x"                                                  # "default"
    config.termination_type = "angle"                                               # "trunk_collision_and_power"
    config.domain_randomization_sampling_type = "none"                              # "step_probability", "none"
    config.domain_randomization_mujoco_model_type = "default"                       # "default", "hard", "none"
    config.domain_randomization_control_type = "default"                            # "default", "hard", "none"
    config.domain_randomization_perturbation_sampling_type = "none"                 # "step_probability", "none"
    config.domain_randomization_perturbation_type = "default"                       # "default", "hard", "none"
    config.observation_noise_type = "none"                                          # "default", "hard", "none"
    config.terrain_type = "plane"                                                   # "plane", "hfield_inverse_pyramid"
    config.trajectory_smoothing_type = "one_euro_filter"                            # "none", "average"
    config.trajectory_smoothing_history_length = 1
    config.add_goal_arrow = False
    config.timestep = 0.005
    config.episode_length_in_seconds = 10
    config.kp = 60.0
    config.kd = 3.0

    return config
