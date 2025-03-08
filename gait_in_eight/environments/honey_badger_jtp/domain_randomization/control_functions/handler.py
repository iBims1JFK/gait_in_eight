from gait_in_eight.environments.honey_badger_jtp.domain_randomization.control_functions.default import DefaultDomainControl
from gait_in_eight.environments.honey_badger_jtp.domain_randomization.control_functions.hard import HardDomainControl
from gait_in_eight.environments.honey_badger_jtp.domain_randomization.control_functions.none import NoneDomainControl


def get_domain_randomization_control_function(name, env, **kwargs):
    if name == "default":
        return DefaultDomainControl(env, **kwargs)
    elif name == "hard":
        return HardDomainControl(env, **kwargs)
    elif name == "none":
        return NoneDomainControl(env, **kwargs)
    else:
        raise NotImplementedError
