from gait_in_eight.environments.honey_badger_cpg.control_functions.pd import PDControl
from gait_in_eight.environments.honey_badger_cpg.control_functions.torque import TorqueControl
from gait_in_eight.environments.honey_badger_cpg.control_functions.inverse_kinematics import InverseKinematicsControl

def get_control_function(name, env, **kwargs):
    if name == "pd":
        return PDControl(env, env.control_frequency_hz, env.kp, env.kd, **kwargs)
    elif name == "torque":
        return TorqueControl(env, **kwargs)
    elif name == "ik":
        return InverseKinematicsControl(env, env.control_frequency_hz, env.kp, env.kd, **kwargs)
    else:
        raise NotImplementedError
