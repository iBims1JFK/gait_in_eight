from ml_collections import config_dict


def get_config(environment_name):
    config = config_dict.ConfigDict()

    config.name = environment_name

    config.seed = 1
    config.async_skip_percentage = 0.0
    config.cycle_cpu_affinity = False
    config.render = False
    config.mode = "train"                                                           # "train", "test"
    config.command_sampling_type = "step_probability"                               # "step_probability", "none", "only_setup"
    config.control_frequency_hz = 50                                                # only used for pd_clip
    config.command_type = "random"                                                  # "random"
    config.termination_type = "angle"                                               # "trunk_collision_and_power"
    config.target_velocity = 1.0                                                    # only used for straight_ahead
    config.reward_type = "default"                                                  # "default"
    config.add_goal_arrow = False
    config.timestep = 0.005
    config.episode_length_in_seconds = 20
    config.nr_envs = 1
    config.trajectory_smoothing_type = "average"                                    # "none", "average"
    config.trajectory_smoothing_history_length = 1
    config.action_space_mode = "default"                                            # "default", "cpg_default" "cpg_frequency", "cpg_residual"
    config.central_pattern_generator_type = "none"                                  # "none", "beat"

    return config
