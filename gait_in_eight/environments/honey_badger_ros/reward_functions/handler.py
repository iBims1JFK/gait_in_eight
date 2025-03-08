from gait_in_eight.environments.honey_badger_ros.reward_functions.simple import SimpleReward
from gait_in_eight.environments.honey_badger_ros.reward_functions.simple2d import SimpleReward2d


def get_reward_function(name, env, **kwargs):
    if name == "simple":
        return SimpleReward(env, **kwargs)
    elif name == "simple2d":
        return SimpleReward2d(env, **kwargs)
    else:
        raise NotImplementedError
