from gait_in_eight.environments.honey_badger_ros.control_functions.inverse_kinematics import InverseKinematics


def get_control_function(name, env, **kwargs):
    if name == "ik":
        return InverseKinematics(env, **kwargs)
    else:
        raise NotImplementedError
