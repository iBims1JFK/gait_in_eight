from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from gait_in_eight.algorithms.redq.default.redq import REDQ
from gait_in_eight.algorithms.redq.default.default_config import get_config
from gait_in_eight.algorithms.redq.default.general_properties import GeneralProperties


REDQ_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(REDQ_FLAX, get_config, REDQ, GeneralProperties)
