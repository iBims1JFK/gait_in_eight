from gait_in_eight.environments.honey_badger_jtp.command_functions.random import RandomCommands
from gait_in_eight.environments.honey_badger_jtp.command_functions.forward import ForwardCommands
from gait_in_eight.environments.honey_badger_jtp.command_functions.curriculum import CurriculumCommands
from gait_in_eight.environments.honey_badger_jtp.command_functions.max_velocity import MaxVelocityCommands


def get_command_function(name, env, **kwargs):
    if name == "random":
        return RandomCommands(env, **kwargs)
    elif name == "forward":
        return ForwardCommands(env, **kwargs)
    elif name == "curriculum":
        return CurriculumCommands(env, **kwargs)
    elif name == "max":
        return MaxVelocityCommands(env, **kwargs)
    else:
        raise NotImplementedError
