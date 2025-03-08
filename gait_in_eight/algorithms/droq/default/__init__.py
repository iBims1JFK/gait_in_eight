from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from gait_in_eight.algorithms.droq.default.droq import DroQ
from gait_in_eight.algorithms.droq.default.default_config import get_config
from gait_in_eight.algorithms.droq.default.general_properties import GeneralProperties


DROQ_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(DROQ_FLAX, get_config, DroQ, GeneralProperties)
