from gait_in_eight.environments.honey_badger_cpg.terrain_functions.plane import PlaneTerrainGeneration


def get_terrain_function(name, env, **kwargs):
    if name == "plane":
        return PlaneTerrainGeneration(env, **kwargs)
    else:
        raise NotImplementedError
