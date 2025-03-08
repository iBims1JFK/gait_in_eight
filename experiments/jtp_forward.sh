python experiment.py \
    --algorithm.name="crossq.default" \
    --algorithm.total_timesteps=2e4 \
    --algorithm.evaluation_frequency=-1 \
    --algorithm.evaluation_episodes=17 \
    --algorithm.determine_fastest_cpu_for_gpu=False \
    --algorithm.device="cpu" \
    --algorithm.learning_rate=0.005 \
    --algorithm.anneal_learning_rate=False \
    --algorithm.batch_renorm_warmup_steps=1000 \
    --algorithm.learning_starts=1000 \
    --algorithm.policy_nr_hidden_units=256 \
    --algorithm.critic_nr_hidden_units=256 \
    --algorithm.buffer_size=1e6 \
    --algorithm.batch_size=128 \
    --algorithm.logging_frequency=1000 \
    --algorithm.target_entropy=-6.0 \
    --environment.control_frequency_hz=40 \
    --environment.name="honey_badger_jtp" \
    --environment.nr_envs=1 \
    --environment.command_type="forward" \
    --environment.reward_type="track_x" \
    --environment.control_type="pd" \
    --environment.async_skip_percentage=0.0 \
    --environment.cycle_cpu_affinity=False \
    --environment.seed=3 \
    --environment.render=False \
    --environment.domain_randomization_perturbation_sampling_type="none" \
    --environment.domain_randomization_sampling_type="none" \
    --environment.observation_noise_type="none" \
    --environment.target_velocity=0.5 \
    --environment.initial_state_type="default" \
    --environment.add_goal_arrow=True \
    --environment.trajectory_smoothing_type="one_euro_filter" \
    --environment.command_sampling_type="only_setup" \
    --environment.trajectory_smoothing_history_length=1 \
    --environment.kp=60.0 \
    --environment.kd=3.0 \
    --runner.mode="train" \
    --runner.track_console=True \
    --runner.track_tb=True \
    --runner.track_wandb=False \
    --runner.save_model=False \
    --runner.wandb_entity="WAND_ENTITY" \
    --runner.project_name="PROJECT_NAME" \
    --runner.exp_name="JTP_FORWARD" \
    --runner.notes=""
