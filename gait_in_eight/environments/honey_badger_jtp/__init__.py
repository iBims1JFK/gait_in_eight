from rl_x.environments.environment_manager import extract_environment_name_from_file, register_environment
from gait_in_eight.environments.honey_badger_jtp.create_env import create_env
from gait_in_eight.environments.honey_badger_jtp.default_config import get_config
from gait_in_eight.environments.honey_badger_jtp.general_properties import GeneralProperties


HONEY_BADGER = extract_environment_name_from_file(__file__)
register_environment(HONEY_BADGER, get_config, create_env, GeneralProperties)
