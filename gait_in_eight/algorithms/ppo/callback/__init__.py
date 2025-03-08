from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from gait_in_eight.algorithms.ppo.callback.ppo import PPO
from gait_in_eight.algorithms.ppo.callback.default_config import get_config
from gait_in_eight.algorithms.ppo.callback.general_properties import GeneralProperties


CALLBACK_PPO = extract_algorithm_name_from_file(__file__)
register_algorithm(CALLBACK_PPO, get_config, PPO, GeneralProperties)
