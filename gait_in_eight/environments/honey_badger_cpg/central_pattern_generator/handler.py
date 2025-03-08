from gait_in_eight.environments.honey_badger_cpg.central_pattern_generator.none import NoGenerator
from gait_in_eight.environments.honey_badger_cpg.central_pattern_generator.beat import BeatGenerator

def get_central_patter_generator_function(name, env, **kwargs):
    if name == "none":
        return NoGenerator(env, **kwargs)
    if name == "beat":
        return BeatGenerator(env, **kwargs)
    else:
        raise NotImplementedError
