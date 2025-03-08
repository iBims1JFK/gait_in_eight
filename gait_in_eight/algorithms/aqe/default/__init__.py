from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from gait_in_eight.algorithms.aqe.default.aqe import AQE
from gait_in_eight.algorithms.aqe.default.default_config import get_config
from gait_in_eight.algorithms.aqe.default.general_properties import GeneralProperties


AQE_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(AQE_FLAX, get_config, AQE, GeneralProperties)
