from gait_in_eight.environments.honey_badger_jtp.trajectory_smoothing_functions.average import AverageSmoothing
from gait_in_eight.environments.honey_badger_jtp.trajectory_smoothing_functions.none import NoSmoothing
from gait_in_eight.environments.honey_badger_jtp.trajectory_smoothing_functions.low_passfilter import LowPassFilter
from gait_in_eight.environments.honey_badger_jtp.trajectory_smoothing_functions.one_euro_filter import OneEuroFilter

def get_trajectory_smoothing_function(name, env, **kwargs):
    if name == "none":
        return NoSmoothing(env, **kwargs)
    if name == "average":
        return AverageSmoothing(env, env.trajectory_smoothing_history_length, **kwargs)
    if name == "low_passfilter":
        return LowPassFilter(env, **kwargs)
    if name == "one_euro_filter":
        return OneEuroFilter(env, **kwargs)
    else:
        raise NotImplementedError
