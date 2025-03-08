from gait_in_eight.environments.honey_badger_cpg.evaluation_functions.default import DefaultEvaluation
from gait_in_eight.environments.honey_badger_cpg.evaluation_functions.yaw import YawEvaluation

def get_evaluation_function(name, env, **kwargs):
    if name == "default":
        return DefaultEvaluation(env, **kwargs)
    elif name == "yaw":
        return YawEvaluation(env, **kwargs)
    else:
        raise NotImplementedError
