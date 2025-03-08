import logging
import gymnasium as gym

from gait_in_eight.environments.honey_badger_ros.environment import HoneyBadgerRos
from gait_in_eight.environments.honey_badger_ros.wrappers import RLXInfo, RecordEpisodeStatistics
from gait_in_eight.environments.honey_badger_ros.general_properties import GeneralProperties
from gait_in_eight.environments.honey_badger_ros.async_vectorized_wrapper import AsyncVectorEnvWithSkipping
from gait_in_eight.environments.honey_badger_ros.cpu_gpu_testing import get_global_cpu_ids, get_fastest_cpu_for_gpu_connection

rlx_logger = logging.getLogger("rl_x")


def create_env(config):
    def make_env(seed, env_cpu_id):
        def thunk():
            env = HoneyBadgerRos(
                seed=seed,
                mode=config.environment.mode,
                control_frequency_hz=config.environment.control_frequency_hz,
                command_sampling_type=config.environment.command_sampling_type,
                command_type=config.environment.command_type,
                target_velocity=config.environment.target_velocity,
                reward_type=config.environment.reward_type,
                timestep=config.environment.timestep,
                termination_type=config.environment.termination_type,
                trajectory_smoothing_type=config.environment.trajectory_smoothing_type,
                trajectory_smoothing_history_length=config.environment.trajectory_smoothing_history_length,
                episode_length_in_seconds=config.environment.episode_length_in_seconds,
                action_space_mode=config.environment.action_space_mode,
                central_pattern_generator_type=config.environment.central_pattern_generator_type,
                cpu_id=env_cpu_id
            )
            env = RecordEpisodeStatistics(env)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk
    
    assert config.environment.nr_envs == 1

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
