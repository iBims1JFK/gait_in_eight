from gait_in_eight.environments.honey_badger_cpg.reward_functions.track_x import TrackXReward
from gait_in_eight.environments.honey_badger_cpg.reward_functions.track_xy import TrackXYReward
from gait_in_eight.environments.honey_badger_cpg.reward_functions.max_x import MaxXReward


def get_reward_function(name, env, **kwargs):
    if name == "track_x":
        return TrackXReward(env, **kwargs)
    elif name == "track_xy":
        return TrackXYReward(env, **kwargs)
    elif name == "max_x":
        return MaxXReward(env, **kwargs)
    else:
        print(name)
        raise NotImplementedError
