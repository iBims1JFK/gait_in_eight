import logging
import gymnasium as gym

from gait_in_eight.environments.honey_badger_jtp.environment import HoneyBadger
from gait_in_eight.environments.honey_badger_jtp.wrappers import RLXInfo, RecordEpisodeStatistics
from gait_in_eight.environments.honey_badger_jtp.general_properties import GeneralProperties
from gait_in_eight.environments.honey_badger_jtp.async_vectorized_wrapper import AsyncVectorEnvWithSkipping
from gait_in_eight.environments.honey_badger_jtp.cpu_gpu_testing import get_global_cpu_ids, get_fastest_cpu_for_gpu_connection

rlx_logger = logging.getLogger("rl_x")


def create_env(config):
    def make_env(seed, env_cpu_id):
        def thunk():
            env = HoneyBadger(
                seed=seed,
                render=config.environment.render,
                mode=config.environment.mode,
                control_type=config.environment.control_type,
                control_frequency_hz=config.environment.control_frequency_hz,
                command_type=config.environment.command_type,
                target_velocity=config.environment.target_velocity,
                command_sampling_type=config.environment.command_sampling_type,
                initial_state_type=config.environment.initial_state_type,
                reward_type=config.environment.reward_type,
                termination_type=config.environment.termination_type,
                domain_randomization_sampling_type=config.environment.domain_randomization_sampling_type,
                domain_randomization_mujoco_model_type=config.environment.domain_randomization_mujoco_model_type,
                domain_randomization_control_type=config.environment.domain_randomization_control_type,
                domain_randomization_perturbation_type=config.environment.domain_randomization_perturbation_type,
                domain_randomization_perturbation_sampling_type=config.environment.domain_randomization_perturbation_sampling_type,
                observation_noise_type=config.environment.observation_noise_type,
                terrain_type=config.environment.terrain_type,
                trajectory_smoothing_type=config.environment.trajectory_smoothing_type,
                trajectory_smoothing_history_length=config.environment.trajectory_smoothing_history_length,
                add_goal_arrow=config.environment.add_goal_arrow,
                timestep=config.environment.timestep,
                episode_length_in_seconds=config.environment.episode_length_in_seconds,
                total_nr_envs=config.environment.nr_envs,
                kp=config.environment.kp,
                kd=config.environment.kd,
                cpu_id=env_cpu_id
            )
            env = RecordEpisodeStatistics(env)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk
    

    global_cpu_ids = None
    if config.environment.cycle_cpu_affinity or config.algorithm.determine_fastest_cpu_for_gpu:
        global_cpu_ids = get_global_cpu_ids()
        rlx_logger.info(f"Global CPU IDs: {global_cpu_ids}")

    fastest_cpu_id = None
    if config.algorithm.determine_fastest_cpu_for_gpu:
        fastest_cpu_id = get_fastest_cpu_for_gpu_connection(global_cpu_ids)

    env_cpu_ids = None
    if config.environment.cycle_cpu_affinity:
        usable_cpu_ids_for_envs = global_cpu_ids.copy()
        if fastest_cpu_id is not None:
            usable_cpu_ids_for_envs.remove(fastest_cpu_id)
        env_cpu_ids = []
        for i in range(config.environment.nr_envs):
            env_cpu_ids.append(usable_cpu_ids_for_envs[i % len(usable_cpu_ids_for_envs)])

    env_list = []
    env_id = 0
    for i in range(config.environment.nr_envs):
        env_cpu_id = None if env_cpu_ids is None else env_cpu_ids[env_id]
        env_list.append(make_env(
            seed=config.environment.seed + i,
            env_cpu_id=env_cpu_id
        ))
        env_id += 1
    if config.environment.nr_envs == 1:
        env = gym.vector.SyncVectorEnv(env_list)
    else:
        env = AsyncVectorEnvWithSkipping(env_list, config.environment.async_skip_percentage)
    env = RLXInfo(env, fastest_cpu_id)
    env.general_properties = GeneralProperties

    env.reset(seed=config.environment.seed)

    return env
