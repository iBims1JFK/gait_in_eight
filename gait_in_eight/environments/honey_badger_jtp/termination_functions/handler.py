from gait_in_eight.environments.honey_badger_jtp.termination_functions.trunk_collision_and_power import TrunkCollisionAndPowerTermination
from gait_in_eight.environments.honey_badger_jtp.termination_functions.no_termination import NoTermination
from gait_in_eight.environments.honey_badger_jtp.termination_functions.angle import AngleTermination


def get_termination_function(name, env, **kwargs):
    if name == "trunk_collision_and_power":
        return TrunkCollisionAndPowerTermination(env, **kwargs)
    elif name == "no_termination":
        return NoTermination(env, **kwargs)
    elif name == "angle":
        return AngleTermination(env, **kwargs)
    else:
        raise NotImplementedError
