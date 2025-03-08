from rl_x.algorithms.algorithm_manager import (
    extract_algorithm_name_from_file,
    register_algorithm,
)
from gait_in_eight.algorithms.sac.utd.sac import SAC
from gait_in_eight.algorithms.sac.utd.default_config import get_config
from gait_in_eight.algorithms.sac.utd.general_properties import GeneralProperties


SAC_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(SAC_FLAX, get_config, SAC, GeneralProperties)
