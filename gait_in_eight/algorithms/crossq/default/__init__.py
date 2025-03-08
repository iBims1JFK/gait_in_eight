from rl_x.algorithms.algorithm_manager import (
    extract_algorithm_name_from_file,
    register_algorithm,
)
from gait_in_eight.algorithms.crossq.default.crossq import CrossQ
from gait_in_eight.algorithms.crossq.default.default_config import get_config
from gait_in_eight.algorithms.crossq.default.general_properties import GeneralProperties


CROSSQ_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(CROSSQ_FLAX, get_config, CrossQ, GeneralProperties)
