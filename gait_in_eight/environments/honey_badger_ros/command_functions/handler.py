from gait_in_eight.environments.honey_badger_ros.command_functions.random import RandomCommands
from gait_in_eight.environments.honey_badger_ros.command_functions.forward import ForwardCommands

def get_command_function(name, env, **kwargs):
    if name == "random":
        return RandomCommands(env, **kwargs)
    elif name == "forward":
        return ForwardCommands(env, **kwargs)
    else:
        raise NotImplementedError

