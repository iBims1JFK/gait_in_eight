import os
import psutil
import logging
import time
from collections import deque
import tree
import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from gait_in_eight.algorithms.ppo.default.general_properties import GeneralProperties
from gait_in_eight.algorithms.ppo.default.policy import get_policy
from gait_in_eight.algorithms.ppo.default.critic import get_critic
from rl_x.algorithms.ppo.flax.batch import Batch

from gait_in_eight.environments import observation_indices as obs_idx

rlx_logger = logging.getLogger("rl_x")


class PPO:
    def __init__(self, config, env, run_path, writer) -> None:
        self.config = config
        self.env = env
        self.writer = writer

        self.save_model = config.runner.save_model
        self.save_path = os.path.join(run_path, "models")
        self.track_console = config.runner.track_console
        self.track_tb = config.runner.track_tb
        self.track_wandb = config.runner.track_wandb
        self.seed = config.environment.seed
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        self.start_learning_rate = config.algorithm.start_learning_rate
        self.end_learning_rate = config.algorithm.end_learning_rate
        self.nr_steps = config.algorithm.nr_steps
        self.nr_epochs = config.algorithm.nr_epochs
        self.minibatch_size = config.algorithm.minibatch_size
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.clip_range = config.algorithm.clip_range
        self.entropy_coef = config.algorithm.entropy_coef
        self.critic_coef = config.algorithm.critic_coef
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.std_dev = config.algorithm.std_dev
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.save_latest_frequency = config.algorithm.save_latest_frequency
        self.determine_fastest_cpu_for_gpu = config.algorithm.determine_fastest_cpu_for_gpu
        self.batch_size = config.environment.nr_envs * config.algorithm.nr_steps
        self.nr_updates = config.algorithm.total_timesteps // self.batch_size
        self.nr_minibatches = self.batch_size // self.minibatch_size

        if self.evaluation_frequency % (self.nr_steps * self.nr_envs) != 0 and self.evaluation_frequency != -1:
            raise ValueError("Evaluation frequency must be a multiple of the number of steps and environments.")
        
        if self.save_latest_frequency % (self.nr_steps * self.nr_envs) != 0 and self.save_model:
            raise ValueError("Save latest frequency must be a multiple of the number of steps and environments.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        if self.determine_fastest_cpu_for_gpu and self.env.fastest_cpu_id is not None:
            p = psutil.Process()
            p.cpu_affinity([self.env.fastest_cpu_id,])
            rlx_logger.info(f"Using fastest CPU for GPU connection: {self.env.fastest_cpu_id}")
        
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key = jax.random.split(self.key, 3)

        self.os_shape = self.env.single_observation_space.shape
        self.as_shape = self.env.single_action_space.shape
        
        self.policy, self.get_processed_action = get_policy(config, self.env)
        self.critic = get_critic(config, self.env)

        self.policy.apply = jax.jit(self.policy.apply)
        self.critic.apply = jax.jit(self.critic.apply)

        def linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
            learning_rate = self.end_learning_rate + fraction * (self.start_learning_rate - self.end_learning_rate)
            return learning_rate

        learning_rate = linear_schedule if self.start_learning_rate != self.end_learning_rate else self.start_learning_rate

        state = jnp.array([self.env.single_observation_space.sample()])

        # Create policy and critic state masks
        self.policy_mask = np.ones(state.shape[-1])
        self.critic_mask = np.ones(state.shape[-1])
        observations_hidden_for_policy_ids = np.concatenate([
            obs_idx.TRUNK_LINEAR_VELOCITIES,
            obs_idx.HEIGHT
        ])
        observations_hidden_for_critic_ids = []
        self.policy_mask[observations_hidden_for_policy_ids] = 0
        self.critic_mask[observations_hidden_for_critic_ids] = 0
        self.policy_mask = jnp.array(self.policy_mask, dtype=bool)
        self.critic_mask = jnp.array(self.critic_mask, dtype=bool)

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, state[:, self.policy_mask]),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, state[:, self.critic_mask]),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf
            self.best_model_file_name = "model_best_jax"
            self.latest_model_file_name = "model_latest_jax"
            best_model_check_point_handler = orbax.checkpoint.PyTreeCheckpointHandler(aggregate_filename=self.best_model_file_name)
            latest_model_check_point_handler = orbax.checkpoint.PyTreeCheckpointHandler(aggregate_filename=self.latest_model_file_name)
            self.best_model_checkpointer = orbax.checkpoint.Checkpointer(best_model_check_point_handler)
            self.latest_model_checkpointer = orbax.checkpoint.Checkpointer(latest_model_check_point_handler)

    
    def train(self):
        @jax.jit
        def get_action_and_value(policy_state: TrainState, critic_state: TrainState, state: np.ndarray, key: jax.random.PRNGKey):
            action_mean, action_logstd = self.policy.apply(policy_state.params, state[:, self.policy_mask])
            action_std = jnp.exp(action_logstd)
            key, subkey = jax.random.split(key)
            action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
            log_prob = -0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
            value = self.critic.apply(critic_state.params, state[:, self.critic_mask])
            processed_action = self.get_processed_action(action)
            return processed_action, action, value.reshape(-1), log_prob.sum(1), key
        

        @jax.jit
        def calculate_gae_advantages(critic_state: TrainState, next_states: np.ndarray, rewards: np.ndarray, terminations: np.ndarray, values: np.ndarray):
            def compute_advantages(carry, t):
                prev_advantage, delta, terminations = carry
                advantage = delta[t] + self.gamma * self.gae_lambda * (1 - terminations[t]) * prev_advantage
                return (advantage, delta, terminations), advantage

            next_values = self.critic.apply(critic_state.params, next_states[:, :, self.critic_mask]).squeeze(-1)
            delta = rewards + self.gamma * next_values * (1.0 - terminations) - values
            init_advantages = delta[-1]
            _, advantages = jax.lax.scan(compute_advantages, (init_advantages, delta, terminations), jnp.arange(self.nr_steps - 2, -1, -1))
            advantages = jnp.concatenate([advantages[::-1], jnp.array([init_advantages])])
            returns = advantages + values
            return advantages, returns
        

        @jax.jit
        def update(policy_state: TrainState, critic_state: TrainState,
                   states: np.ndarray, actions: np.ndarray, advantages: np.ndarray, returns: np.ndarray, values: np.ndarray, log_probs: np.ndarray,
                   key: jax.random.PRNGKey):
            def loss_fn(policy_params, critic_params, state_b, action_b, log_prob_b, return_b, advantage_b):
                # Policy loss
                action_mean, action_logstd = self.policy.apply(policy_params, state_b[self.policy_mask])
                action_std = jnp.exp(action_logstd)
                new_log_prob = -0.5 * ((action_b - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
                new_log_prob = new_log_prob.sum(1)
                entropy = action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
                
                logratio = new_log_prob - log_prob_b
                ratio = jnp.exp(logratio)
                approx_kl_div = (ratio - 1) - logratio
                clip_fraction = jnp.float32((jnp.abs(ratio - 1) > self.clip_range))

                pg_loss1 = -advantage_b * ratio
                pg_loss2 = -advantage_b * jnp.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)
                pg_loss = jnp.maximum(pg_loss1, pg_loss2)
                
                entropy_loss = entropy.sum(1)
                
                # Critic loss
                new_value = self.critic.apply(critic_params, state_b[self.critic_mask])
                critic_loss = 0.5 * (new_value - return_b) ** 2

                # Combine losses
                loss = pg_loss - self.entropy_coef * entropy_loss + self.critic_coef * critic_loss

                # Create metrics
                metrics = {
                    "loss/policy_gradient_loss": pg_loss,
                    "loss/critic_loss": critic_loss,
                    "loss/entropy_loss": entropy_loss,
                    "policy_ratio/approx_kl": approx_kl_div,
                    "policy_ratio/clip_fraction": clip_fraction,
                }

                return loss, (metrics)
            

            batch_states = states.reshape((-1,) + self.os_shape)
            batch_actions = actions.reshape((-1,) + self.as_shape)
            batch_advantages = advantages.reshape(-1)
            batch_returns = returns.reshape(-1)
            batch_log_probs = log_probs.reshape(-1)

            vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, 0, 0, 0, 0, 0), out_axes=0)
            safe_mean = lambda x: jnp.mean(x) if x is not None else x
            mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
            grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1), has_aux=True)

            key, subkey = jax.random.split(key)
            batch_indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs, 1))
            batch_indices = jax.random.permutation(subkey, batch_indices, axis=1, independent=True)
            batch_indices = batch_indices.reshape((self.nr_epochs * self.nr_minibatches, self.minibatch_size))

            def minibatch_update(carry, minibatch_indices):
                policy_state, critic_state = carry

                minibatch_advantages = batch_advantages[minibatch_indices]
                minibatch_advantages = (minibatch_advantages - jnp.mean(minibatch_advantages)) / (jnp.std(minibatch_advantages) + 1e-8)

                (loss, (metrics)), (policy_gradients, critic_gradients) = grad_loss_fn(
                    policy_state.params,
                    critic_state.params,
                    batch_states[minibatch_indices],
                    batch_actions[minibatch_indices],
                    batch_log_probs[minibatch_indices],
                    batch_returns[minibatch_indices],
                    minibatch_advantages
                )

                policy_state = policy_state.apply_gradients(grads=policy_gradients)
                critic_state = critic_state.apply_gradients(grads=critic_gradients)

                metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
                metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)

                carry = (policy_state, critic_state)

                return carry, (metrics)
            
            init_carry = (policy_state, critic_state)
            carry, (metrics) = jax.lax.scan(minibatch_update, init_carry, batch_indices)
            policy_state, critic_state = carry

            # Calculate mean metrics
            mean_metrics = {key: jnp.mean(metrics[key]) for key in metrics}
            mean_metrics["lr/learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
            mean_metrics["v_value/explained_variance"] = 1 - jnp.var(returns - values) / (jnp.var(returns) + 1e-8)
            mean_metrics["policy/std_dev"] = jnp.mean(jnp.exp(policy_state.params["params"]["policy_logstd"]))

            return policy_state, critic_state, mean_metrics, key


        @jax.jit
        def get_deterministic_eval_action(policy_state: TrainState, state: np.ndarray):
            action_mean, action_logstd = self.policy.apply(policy_state.params, state[:, self.eval_policy_mask])
            return self.get_processed_action(action_mean)
        

        self.set_train_mode()

        batch = Batch(
            states=np.zeros((self.nr_steps, self.nr_envs) + self.os_shape),
            next_states=np.zeros((self.nr_steps, self.nr_envs) + self.os_shape),
            actions=np.zeros((self.nr_steps, self.nr_envs) + self.as_shape),
            rewards=np.zeros((self.nr_steps, self.nr_envs)),
            values=np.zeros((self.nr_steps, self.nr_envs)),
            terminations=np.zeros((self.nr_steps, self.nr_envs)),
            log_probs=np.zeros((self.nr_steps, self.nr_envs)),
            advantages=np.zeros((self.nr_steps, self.nr_envs)),
            returns=np.zeros((self.nr_steps, self.nr_envs)),
        )

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        state, _ = self.env.reset()
        global_step = 0
        nr_updates = 0
        nr_episodes = 0
        steps_metrics = {}
        while global_step < self.total_timesteps:
            start_time = time.time()
            time_metrics = {}


            # Acting
            dones_this_rollout = 0
            step_info_collection = {}
            for step in range(self.nr_steps):
                processed_action, action, value, log_prob, self.key = get_action_and_value(self.policy_state, self.critic_state, state, self.key)
                next_state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
                done = terminated | truncated
                actual_next_state = next_state.copy()
                for i, single_done in enumerate(done):
                    if single_done:
                        actual_next_state[i] = self.env.get_final_observation_at_index(info, i)
                        saving_return_buffer.append(self.env.get_final_info_value_at_index(info, "episode_return", i))
                        dones_this_rollout += 1
                for key, info_value in self.env.get_logging_info_dict(info).items():
                    step_info_collection.setdefault(key, []).extend(info_value)

                batch.states[step] = state
                batch.next_states[step] = actual_next_state
                batch.actions[step] = action
                batch.rewards[step] = reward
                batch.values[step] = value
                batch.terminations[step] = terminated
                batch.log_probs[step] = log_prob
                state = next_state
                global_step += self.nr_envs
            nr_episodes += dones_this_rollout
            
            acting_end_time = time.time()
            time_metrics["time/acting_time"] = acting_end_time - start_time


            # Calculating advantages and returns
            batch.advantages, batch.returns = calculate_gae_advantages(self.critic_state, batch.next_states, batch.rewards, batch.terminations, batch.values)
            
            calc_adv_return_end_time = time.time()
            time_metrics["time/calc_adv_and_return_time"] = calc_adv_return_end_time - acting_end_time


            # Optimizing
            self.policy_state, self.critic_state, optimization_metrics, self.key = update(
                self.policy_state, self.critic_state,
                batch.states, batch.actions, batch.advantages, batch.returns, batch.values, batch.log_probs,
                self.key
            )
            optimization_metrics = {key: value.item() for key, value in optimization_metrics.items()}
            nr_updates += self.nr_epochs * self.nr_minibatches

            optimizing_end_time = time.time()
            time_metrics["time/optimizing_time"] = optimizing_end_time - calc_adv_return_end_time


            # Evaluating
            evaluation_metrics = {}
            if global_step % self.evaluation_frequency == 0 and self.evaluation_frequency != -1:
                self.set_eval_mode()
                state, _ = self.eval_env.reset()
                eval_nr_episodes = 0
                evaluation_metrics = {
                    "eval/episode_return": [],
                    "eval/episode_length": [],
                    "eval/track_perf_perc": [],
                    "eval/eps_track_perf_perc": []
                }
                while True:
                    processed_action = get_deterministic_eval_action(self.policy_state, state)
                    state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
                    for key, info_value in self.env.get_logging_info_dict(info).items():
                        if "track_perf_perc" in key:
                            name = key.replace("env_info", "eval")
                            evaluation_metrics[name].extend(info_value)
                    done = terminated | truncated
                    for i, single_done in enumerate(done):
                        if single_done:
                            eval_nr_episodes += 1
                            evaluation_metrics["eval/episode_return"].append(
                                self.eval_env.get_final_info_value_at_index(info, "episode_return", i)
                            )
                            evaluation_metrics["eval/episode_length"].append(
                                self.eval_env.get_final_info_value_at_index(info, "episode_length", i)
                            )
                            evaluation_metrics["eval/track_perf_perc"].append(
                                self.eval_env.get_final_info_value_at_index(info, "env_info/track_perf_perc", i)
                            )
                            break
                    if np.all(eval_nr_episodes == self.evaluation_episodes):
                        break
                evaluation_metrics = {key: np.mean(value) for key, value in evaluation_metrics.items()}
                self.set_train_mode()
            
            evaluating_end_time = time.time()
            time_metrics["time/evaluating_time"] = evaluating_end_time - optimizing_end_time
            

            # Saving
            # Also only save when there were finished episodes this update
            if self.save_model and dones_this_rollout > 0:
                mean_return = np.mean(saving_return_buffer)
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save("best")
            if self.save_model and global_step % self.save_latest_frequency == 0:
                self.save("latest")
            
            saving_end_time = time.time()
            time_metrics["time/saving_time"] = saving_end_time - evaluating_end_time

            time_metrics["time/sps"] = int((self.nr_steps * self.nr_envs) / (saving_end_time - start_time))


            # Logging
            self.start_logging(global_step)

            steps_metrics["steps/nr_env_steps"] = global_step
            steps_metrics["steps/nr_updates"] = nr_updates
            steps_metrics["steps/nr_episodes"] = nr_episodes

            rollout_info_metrics = {}
            env_info_metrics = {}
            if step_info_collection:
                info_names = list(step_info_collection.keys())
                for info_name in info_names:
                    metric_group = "rollout" if info_name in ["episode_return", "episode_length"] else "env_info"
                    metric_dict = rollout_info_metrics if metric_group == "rollout" else env_info_metrics
                    mean_value = np.mean(step_info_collection[info_name])
                    if mean_value == mean_value:  # Check if mean_value is NaN
                        metric_dict[f"{metric_group}/{info_name}"] = mean_value
            
            combined_metrics = {**rollout_info_metrics, **evaluation_metrics, **env_info_metrics, **steps_metrics, **time_metrics, **optimization_metrics}
            for key, value in combined_metrics.items():
                self.log(f"{key}", value, global_step)

            self.end_logging()


    def log(self, name, value, step):
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_console:
            self.log_console(name, value)
    

    def log_console(self, name, value):
        value = np.format_float_positional(value, trim="-")
        rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │", flush=False)


    def start_logging(self, step):
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐", flush=False)
        else:
            rlx_logger.info(f"Step: {step}")


    def end_logging(self):
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")

    
    def save(self, type):
        checkpoint = {
            "config_algorithm": self.config.algorithm.to_dict(),
            "policy": self.policy_state,
            "critic": self.critic_state
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        if type == "best":
            self.best_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
            os.rename(f"{self.save_path}/tmp/{self.best_model_file_name}", f"{self.save_path}/{self.best_model_file_name}")
            os.remove(f"{self.save_path}/tmp/_METADATA")
            os.rmdir(f"{self.save_path}/tmp")

            if self.track_wandb:
                wandb.save(f"{self.save_path}/{self.best_model_file_name}", base_path=self.save_path)
        elif type == "latest":
            self.latest_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
            os.rename(f"{self.save_path}/tmp/{self.latest_model_file_name}", f"{self.save_path}/{self.latest_model_file_name}")
            os.remove(f"{self.save_path}/tmp/_METADATA")
            os.rmdir(f"{self.save_path}/tmp")

            if self.track_wandb:
                wandb.save(f"{self.save_path}/{self.latest_model_file_name}", base_path=self.save_path)


    def load(config, env, run_path, writer, explicitly_set_algorithm_params):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = "/".join(splitted_path[:-1])
        checkpoint_file_name = splitted_path[-1]

        check_point_handler = orbax.checkpoint.PyTreeCheckpointHandler(aggregate_filename=checkpoint_file_name)
        checkpointer = orbax.checkpoint.Checkpointer(check_point_handler)

        loaded_algorithm_config = checkpointer.restore(checkpoint_dir)["config_algorithm"]
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params:
                config.algorithm[key] = value
        model = PPO(config, env, run_path, writer)

        target = {
            "config_algorithm": config.algorithm.to_dict(),
            "policy": model.policy_state,
            "critic": model.critic_state
        }
        checkpoint = checkpointer.restore(checkpoint_dir, item=target)

        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]

        return model
    

    def test(self, episodes):
        @jax.jit
        def get_action(policy_state: TrainState, state: np.ndarray):
            action_mean, action_logstd = self.policy.apply(policy_state.params, state[:, self.policy_mask])
            return self.get_processed_action(action_mean)
        
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0
            state, _ = self.env.reset()
            while not done:
                processed_action = get_action(self.policy_state, state)
                state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
                done = terminated | truncated
                episode_return += reward
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")
    
            
    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...


    def general_properties():
        return GeneralProperties
