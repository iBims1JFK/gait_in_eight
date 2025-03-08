from gait_in_eight.environments.honey_badger_jtp.sampling_functions.step_probability import StepProbabilitySampling
from gait_in_eight.environments.honey_badger_jtp.sampling_functions.none import NoneSampling
from gait_in_eight.environments.honey_badger_ros.sampling_functions.only_setup import OnlySetupSampling


def get_sampling_function(name, env, **kwargs):
    if name == "step_probability":
        return StepProbabilitySampling(env, **kwargs)
    elif name == "none":
        return NoneSampling(env, **kwargs)
    elif name == "only_setup":
        return OnlySetupSampling(env, **kwargs)
    else:
        raise NotImplementedError
