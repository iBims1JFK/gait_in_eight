from gait_in_eight.environments.honey_badger_cpg.domain_randomization.perturbation_functions.default import DefaultDomainPerturbation
from gait_in_eight.environments.honey_badger_cpg.domain_randomization.perturbation_functions.hard import HardDomainPerturbation
from gait_in_eight.environments.honey_badger_cpg.domain_randomization.perturbation_functions.none import NoneDomainPerturbation


def get_domain_randomization_perturbation_function(name, env, **kwargs):
    if name == "default":
        return DefaultDomainPerturbation(env, **kwargs)
    elif name == "hard":
        return HardDomainPerturbation(env, **kwargs)
    elif name == "none":
        return NoneDomainPerturbation(env, **kwargs)
    else:
        raise NotImplementedError
