from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return (
            Policy(
                env.single_action_space.shape,
                config.algorithm.std_dev,
                config.algorithm.policy_mean_abs_clip,
                config.algorithm.policy_std_min_clip, config.algorithm.policy_std_max_clip
            ),
            get_processed_action_function()
            )


class Policy(nn.Module):
    as_shape: Sequence[int]
    std_dev: float
    policy_mean_abs_clip: float
    policy_std_min_clip: float
    policy_std_max_clip: float

    @nn.compact
    def __call__(self, x):
        policy_mean = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        policy_mean = nn.LayerNorm()(policy_mean)
        policy_mean = nn.elu(policy_mean)
        policy_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(policy_mean)
        policy_mean = nn.elu(policy_mean)
        policy_mean = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(policy_mean)
        policy_mean = nn.elu(policy_mean)
        policy_mean = nn.Dense(np.prod(self.as_shape).item(), kernel_init=orthogonal(0.01), bias_init=constant(0.0))(policy_mean)
        policy_mean = jnp.clip(policy_mean, -self.policy_mean_abs_clip, self.policy_mean_abs_clip)
        policy_logstd = self.param("policy_logstd", constant(jnp.log(self.std_dev)), (1, np.prod(self.as_shape).item()))
        policy_logstd = jnp.clip(policy_logstd, jnp.log(self.policy_std_min_clip), jnp.log(self.policy_std_max_clip))
        return policy_mean, policy_logstd


def get_processed_action_function():
    def get_clipped_and_scaled_action(action):
        return action
    return jax.jit(get_clipped_and_scaled_action)
