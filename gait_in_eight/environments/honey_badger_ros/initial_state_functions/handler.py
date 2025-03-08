from gait_in_eight.environments.honey_badger_ros.initial_state_functions.random import RandomInitialState
from gait_in_eight.environments.honey_badger_ros.initial_state_functions.ros import RosDefaultInitialState


def get_initial_state_function(name, env, **kwargs):
    if name == "default":
        return RosDefaultInitialState(env, **kwargs)
    elif name == "random":
        return RandomInitialState(env, **kwargs)
    else:
        raise NotImplementedError
