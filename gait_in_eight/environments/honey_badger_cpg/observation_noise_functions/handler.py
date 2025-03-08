from gait_in_eight.environments.honey_badger_cpg.observation_noise_functions.default import DefaultObservationNoise
from gait_in_eight.environments.honey_badger_cpg.observation_noise_functions.hard import HardObservationNoise
from gait_in_eight.environments.honey_badger_cpg.observation_noise_functions.none import NoneObservationNoise


def get_observation_noise_function(name, env, **kwargs):
    if name == "default":
        return DefaultObservationNoise(env, **kwargs)
    elif name == "hard":
        return HardObservationNoise(env, **kwargs)
    elif name == "none":
        return NoneObservationNoise(env, **kwargs)
    else:
        raise NotImplementedError
