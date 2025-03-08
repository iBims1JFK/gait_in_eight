from gait_in_eight.environments.honey_badger_jtp.control_functions.pd import PDControl
from gait_in_eight.environments.honey_badger_jtp.control_functions.torque import TorqueControl


def get_control_function(name, env, **kwargs):
    if name == "pd":
        return PDControl(env, **kwargs)
    elif name == "torque":
        return TorqueControl(env, **kwargs)
    else:
        raise NotImplementedError
