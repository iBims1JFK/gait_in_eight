from pathlib import Path
import psutil
import mujoco 
import numpy as np
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
from dm_control import mjcf
import pygame
import rclpy.executors
import threading

import time
import sys
from gait_in_eight.environments.honey_badger_ros.ros_connection.remote import Remote


# from gait_in_eight.environments.honey_badger_ros.viewer import MujocoViewer
from gait_in_eight.environments.honey_badger_ros.control_functions.handler import get_control_function
from gait_in_eight.environments.honey_badger_ros.command_functions.handler import get_command_function
from gait_in_eight.environments.honey_badger_ros.sampling_functions.handler import get_sampling_function
from gait_in_eight.environments.honey_badger_ros.initial_state_functions.handler import get_initial_state_function
from gait_in_eight.environments.honey_badger_ros.reward_functions.handler import get_reward_function
from gait_in_eight.environments.honey_badger_ros.termination_functions.handler import get_termination_function
# from gait_in_eight.environments.honey_badger_ros.domain_randomization.mujoco_model_functions.handler import get_domain_randomization_mujoco_model_function
# from gait_in_eight.environments.honey_badger_ros.domain_randomization.control_functions.handler import get_domain_randomization_control_function
# from gait_in_eight.environments.honey_badger_ros.domain_randomization.perturbation_functions.handler import get_domain_randomization_perturbation_function
# from gait_in_eight.environments.honey_badger_ros.observation_noise_functions.handler import get_observation_noise_function
from gait_in_eight.environments.honey_badger_ros.terrain_functions.handler import get_terrain_function
from gait_in_eight.environments.honey_badger_ros.ros_connection.ros_connection import RosConnectionNode
from gait_in_eight.environments.honey_badger_ros.trajectory_smoothing_functions.handler import get_trajectory_smoothing_function
from gait_in_eight.environments.honey_badger_ros.central_pattern_generator.handler import get_central_patter_generator_function


class HoneyBadgerRos(gym.Env):
    def __init__(self, seed,
                 mode,
                 control_frequency_hz, command_sampling_type, command_type, target_velocity, 
                 reward_type, timestep, episode_length_in_seconds, termination_type,
                 trajectory_smoothing_type,
                 trajectory_smoothing_history_length,
                 action_space_mode, central_pattern_generator_type,
                 cpu_id=None):
        
        if cpu_id is not None:
            p = psutil.Process()
            p.cpu_affinity([cpu_id,])
        
        if action_space_mode == "default":
            from gait_in_eight.environments import observation_indices_wo_feet as obs_idx
        else:
            from gait_in_eight.environments import observation_indices_wo_feet_cpg as obs_idx
        
        self.obs_idx = obs_idx
            

        # self.seed = seed
        self.trajectory_smoothing_history_length = trajectory_smoothing_history_length

        self.mode = mode
        self.eval = False
        self.eval_at_last_setup = self.eval
        self.np_rng = np.random.default_rng(seed)
        self.total_nr_envs = 1
        self.target_velocity = target_velocity
        self.control_frequency_hz = control_frequency_hz
        self.action_space_mode = action_space_mode
        self.central_pattern_generator_type = central_pattern_generator_type

        # self.nominal_joint_positions = np.array([
        #     -0.05, 0.7, -1.4,
        #     0.05, -0.7, 1.4,
        #     -0.05, -0.7, 1.4,
        #     0.05, 0.7, -1.4,
        # ])


        self.nominal_joint_positions = np.array([
            -0.1, 0.6, -1.2,
            0.1, -0.6, 1.2,
            -0.1, -0.6, 1.2,
            0.1, 0.6, -1.2,
        ])

        self.max_joint_velocities = np.array([  
            25.0, 25.0, 25.0,
            25.0, 25.0, 25.0,
            25.0, 25.0, 25.0,
            25.0, 25.0, 25.0
        ])

        self.total_timesteps = 0
        self.goal_x_velocity = 0
        self.goal_y_velocity = 0
        self.goal_yaw_velocity = 0

        self.simulation_step = 0
        self.termination_function = get_termination_function(termination_type, self)
        self.terrain_function = get_terrain_function("plane", self)
        xml_file_name = self.terrain_function.xml_file_name
        xml_path = (Path(__file__).resolve().parent.parent / "data" / xml_file_name).as_posix()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.episode_length_in_seconds = episode_length_in_seconds
        # if mode == "test":
        #     initial_state_type = "default"
        #     domain_randomization_sampling_type = "none"
        #     domain_randomization_perturbation_sampling_type = "none"
        #     observation_noise_type = "none"

        self.nr_substeps = int(round(1 / self.control_frequency_hz / timestep))
        self.nr_intermediate_steps = 1
        self.dt = timestep * self.nr_substeps * self.nr_intermediate_steps
        self.horizon = int(round(episode_length_in_seconds * self.control_frequency_hz))
        self.command_sampling_function = get_sampling_function(command_sampling_type, self)
        self.command_function = get_command_function(command_type, self)
        self.reward_function = get_reward_function(reward_type, self)
        self.initial_state_function = get_initial_state_function("default", self)
        self.initial_observation = self.get_initial_observation()
        self.trajectory_smoothing_function = get_trajectory_smoothing_function(trajectory_smoothing_type, self)
        self.central_pattern_generator_function = get_central_patter_generator_function(central_pattern_generator_type, self)
        # control function that computes target joint positions
        self.control_function = get_control_function("ik", self)

        # changed single_action_space
        # action_space_low = -np.ones(self.model.nu) * np.Inf
        # action_space_high = np.ones(self.model.nu) * np.Inf
        if self.action_space_mode == "default":
            action_space_low = -np.asarray([0.2, 0.4, 0.4] * 4)
            action_space_high = np.asarray([0.2, 0.4, 0.4] * 4) 
        elif self.action_space_mode == "cpg_default":
            # x and y maybe flipped 
            action_space_low = -np.asarray([0.15, 0.15, 0.10] * 4)
            action_space_high = np.asarray([0.15, 0.15, 0.10] * 4)
        elif self.action_space_mode == "cpg_frequency":
            action_space_low = -np.asarray([*([0.15, 0.15, 0.10] * 4), -1])
            action_space_high = np.asarray([*([0.15, 0.15, 0.10] * 4), 4])
        elif self.action_space_mode == "cpg_residual":
            action_space_low = -np.asarray([0.15, 0.15, 0.10, 0.1, 0.2, 0.2] * 4)
            action_space_high = np.asarray([0.15, 0.15, 0.10, 0.1, 0.2, 0.2] * 4)
        self.action_space = gym.spaces.Box(low=action_space_low, high=action_space_high, dtype=np.float32)

        self.joint_order = [
            self.obs_idx.QUADRUPED_BACK_LEFT_HIP, self.obs_idx.QUADRUPED_BACK_LEFT_THIGH, self.obs_idx.QUADRUPED_BACK_LEFT_CALF,
            self.obs_idx.QUADRUPED_BACK_RIGHT_HIP, self.obs_idx.QUADRUPED_BACK_RIGHT_THIGH, self.obs_idx.QUADRUPED_BACK_RIGHT_CALF,
            self.obs_idx.QUADRUPED_FRONT_RIGHT_HIP, self.obs_idx.QUADRUPED_FRONT_RIGHT_THIGH, self.obs_idx.QUADRUPED_FRONT_RIGHT_CALF,
            self.obs_idx.QUADRUPED_FRONT_LEFT_HIP, self.obs_idx.QUADRUPED_FRONT_LEFT_THIGH, self.obs_idx.QUADRUPED_FRONT_LEFT_CALF
        ]
        # self.feet_order = [
        #     self.obs_idx.QUADRUPED_BACK_LEFT_FOOT, self.obs_idx.QUADRUPED_BACK_RIGHT_FOOT, self.obs_idx.QUADRUPED_FRONT_RIGHT_FOOT, self.obs_idx.QUADRUPED_FRONT_LEFT_FOOT
        # ]

        # self.foot_names = ["RL_foot", "RR_foot", "FR_foot", "FL_foot"]
        self.joint_names = [
            "rl_j0", "rl_j1", "rl_j2",
            "rr_j0", "rr_j1", "rr_j2",
            "fr_j0", "fr_j1", "fr_j2",
            "fl_j0", "fl_j1", "fl_j2",
        ]

        self.remote = Remote(self.mode)
        rclpy.init()
        self.connection = RosConnectionNode(self.joint_names, self.nominal_joint_positions, self, self.remote)
        # dont use multi threaded executor, it is implemented poorly
        ex = rclpy.executors.SingleThreadedExecutor()
        ex.add_node(self.connection)
        thread = threading.Thread(target=ex.spin, daemon=False)
        thread.name = "ros_connection"
        thread.start()
        self.updates = 0
        self.update_time = time.perf_counter()

        qpos, qvel, qacc = get_initial_state_function("default", self).setup()
        self.data.qpos = qpos
        self.data.qvel = qvel
        self.data.qacc = qacc

        self.update_orientation_attributes()
        
        self.observation_space = self.get_observation_space()

        self.current_action = np.zeros(self.model.nu)
        self.initial_observation = self.get_initial_observation()

        self.reward_function.init()
        self.trajectory_smoothing_function.init()

        # if self.mode == "test":
        #     pygame.init()
        #     pygame.joystick.init()
        #     self.joystick_present = False
        #     if pygame.joystick.get_count() > 0:
        #         self.joystick = pygame.joystick.Joystick(0)
        #         self.joystick.init()
        #         self.joystick_present = True


    def reset(self, seed=None):
        # if not self.simulation_step == 0 and self.episode_step < self.horizon:
        if (not self.simulation_step == 0 and self.episode_step < self.horizon) or self.remote.is_reset():
            while self.remote.is_reset():
                time.sleep(0.1)
        # self.connection.reset_qvel()
        self.episode_step = 0
        self.current_action = np.zeros(self.model.nu)
        if self.action_space_mode == "default":
            self.last_action = np.zeros(self.model.nu)
        elif self.action_space_mode == "cpg_default":
            self.last_action = np.zeros(self.model.nu)
        elif self.action_space_mode == "cpg_frequency":
            self.last_action = np.zeros(self.model.nu + 1)
        elif self.action_space_mode == "cpg_residual":
            self.last_action = np.zeros(self.model.nu + 4 * 3)
        self.current_torques = np.zeros(self.model.nu)
        self.termination_function.setup()
        self.central_pattern_generator_function.setup()
        self.cpg = np.array([0, np.pi])
        self.trajectory_smoothing_function.init()


        self.command_function.setup()
        self.reward_function.setup()
        qpos, qvel, qacc = self.initial_state_function.setup()

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.qacc[:] = qacc

        self.update_orientation_attributes()

        return self.get_observation(), {}


    def step(self, action):
        while not self.remote.get_state() == self.remote.LEARN:
            time.sleep(0.1)


                
        explicit_commands = False
        if self.mode == "test":
            if self.remote:
                try:
                    self.goal_x_velocity = float(self.remote.goal_x_velocity)
                    self.goal_y_velocity = float(self.remote.goal_y_velocity)
                    self.goal_yaw_velocity = float(self.remote.goal_yaw_velocity)
                except:
                    self.goal_x_velocity = 0
                    self.goal_y_velocity = 0
                    self.goal_yaw_velocity = 0
                explicit_commands = True
            elif Path("commands.txt").is_file():
                with open("commands.txt", "r") as f:
                    commands = f.readlines()
                if len(commands) == 3:
                    self.goal_x_velocity = float(commands[0])
                    self.goal_y_velocity = float(commands[1])
                    self.goal_yaw_velocity = float(commands[2])
                    explicit_commands = True

        # if not explicit_commands:
        #     if self.total_timesteps == 0:
        #         self.goal_x_velocity, self.goal_y_velocity, self.goal_yaw_velocity = self.command_function.get_next_command()
        #         self.start_time = time.time()
        if not explicit_commands:
            should_sample_commands = self.command_sampling_function.step()
            if should_sample_commands or self.total_timesteps == 0:
                self.goal_x_velocity, self.goal_y_velocity, self.goal_yaw_velocity = self.command_function.get_next_command()
                self.remote.set_train_goal(self.goal_x_velocity, self.goal_y_velocity, self.goal_yaw_velocity)

        intermediate_time = time.perf_counter() - self.update_time
        if intermediate_time < 0.008:
            time.sleep(0.008 - intermediate_time)

        if self.action_space_mode == "default":
            action = self.trajectory_smoothing_function.step(action)
            action = action + self.nominal_joint_positions
        # elif self.action_space_mode == "cpg_default":
        #     offset = self.central_pattern_generator_function.step()
        #     action = self.control_function.process_action(action, offset)
        #     self.cpg = offset
        # elif self.action_space_mode == "cpg_frequency":
        #     offset = self.central_pattern_generator_function.step(action[-1])
        #     action = self.control_function.process_action(action, offset)
        #     self.cpg = offset
        # elif self.action_space_mode == "cpg_residual":
        #     # action_copy = action.copy()
        #     action = action.reshape(-1, 6)
        #     action_ik = action[:, 3:]
        #     action_ik = action_ik.flatten()
        #     action_ik += self.nominal_joint_positions
        #     action = action[:, :3]
        #     action = action.flatten()
        #     offset = self.central_pattern_generator_function.step()
        #     action = self.control_function.process_action(action, offset)
        #     action += action_ik
        #     self.cpg = offset

        if self.action_space_mode == "default":
            self.connection.send_joint_command(action)
        else:
            self.cpg = self.connection.cpg.get_offset()
            self.connection.set_action(action)
        time_diff = time.perf_counter() - self.update_time

        waiting_time = (1/self.control_frequency_hz - time_diff) if 1/self.control_frequency_hz > time_diff else 0
        waiting_time = max(waiting_time, 0)
        if time_diff > 1/self.control_frequency_hz:
            print(f"Time difference too high: {time_diff},{waiting_time}")
        # time.sleep(waiting_time)
        self.precise_sleeping(waiting_time)
        self.update_time = time.perf_counter()
        self.updates += 1

        qpos, qvel, qacc, ctrl, april_qvel = self.connection.get_data()
        self.data.qpos = qpos
        self.data.qvel = qvel
        self.data.qacc = qacc
        self.data.qvel[6:] = np.clip(
                qvel[6:],
                -self.max_joint_velocities,
                self.max_joint_velocities,
        )
        self.data.ctrl = ctrl

        self.simulation_step += self.nr_substeps

        self.update_orientation_attributes()
        
        self.current_action = action.copy()
        # self.current_torques = torques

        next_observation = self.get_observation()
        terminated = self.termination_function.should_terminate(next_observation)
        if terminated:
            self.remote.set_state(Remote.COLLAPSE)
        terminated = terminated | self.remote.is_reset()
        truncated = self.episode_step + 1 >= self.horizon
        done = terminated | truncated
        reward, r_info = self.get_reward_and_info(done)
        info = {**r_info}
        info["simulation_step"] = self.simulation_step
        if april_qvel is not None:
            info["april_qvel"] = april_qvel

        self.reward_function.step(action)
        self.command_function.step(next_observation, reward, done, info)
        self.initial_state_function.step(next_observation, reward, done, info)
        
        self.last_action = action.copy()
        self.episode_step += 1
        if not self.eval:
            self.total_timesteps += 1
        end_time = time.time()
        return next_observation, reward, terminated, truncated, info


    def update_orientation_attributes(self):
        self.orientation_quat = R.from_quat([self.data.qpos[4], self.data.qpos[5], self.data.qpos[6], self.data.qpos[3]])
        self.orientation_euler = self.orientation_quat.as_euler("xyz")
        self.orientation_quat_inv = self.orientation_quat.inv()


    def get_observation_space(self):
        space_low = np.array([-np.inf] * self.obs_idx.OBSERVATION_SIZE, dtype=np.float32)
        space_high = np.array([np.inf] * self.obs_idx.OBSERVATION_SIZE, dtype=np.float32)

        return gym.spaces.Box(low=space_low, high=space_high, shape=space_low.shape, dtype=np.float32)


    def get_initial_observation(self):
        return np.zeros(self.obs_idx.OBSERVATION_SIZE, dtype=np.float32)
    

    def get_observation(self):
        observation = self.initial_observation.copy()

        # Dynamic observations
        for i, joint_range in enumerate(self.joint_order):
            observation[joint_range[0]] = self.data.qpos[i+7] - self.nominal_joint_positions[i]
            observation[joint_range[1]] = self.data.qvel[i+6]
            observation[joint_range[2]] = self.current_action[i]

        # for i, foot_range in enumerate(self.feet_order):
        #     foot_name = self.foot_names[i]
        #     observation[foot_range[0]] = self.check_collision("floor", foot_name)
        #     if foot_name == "FL_foot":
        #         observation[foot_range[1]] = self.reward_function.time_since_last_touchdown_fl
        #     elif foot_name == "FR_foot":
        #         observation[foot_range[1]] = self.reward_function.time_since_last_touchdown_fr
        #     elif foot_name == "RL_foot":
        #         observation[foot_range[1]] = self.reward_function.time_since_last_touchdown_rl
        #     elif foot_name == "RR_foot":
        #         observation[foot_range[1]] = self.reward_function.time_since_last_touchdown_rr
        

        # IMU accelerations
        trunk_linear_acceleration = self.orientation_quat_inv.apply(self.data.qacc[:3])
        observation[self.obs_idx.TRUNK_LINEAR_ACCELERATIONS] = trunk_linear_acceleration

        trunk_angular_acceleration = self.data.qacc[3:6]
        observation[self.obs_idx.TRUNK_ANGULAR_ACCELERATIONS] = trunk_angular_acceleration
        

        # General observations
        trunk_linear_velocity = self.data.qvel[:3]
        observation[self.obs_idx.TRUNK_LINEAR_VELOCITIES] = trunk_linear_velocity

        trunk_angular_velocity = self.data.qvel[3:6]
        observation[self.obs_idx.TRUNK_ANGULAR_VELOCITIES] = trunk_angular_velocity

        goal_velocity = np.array([self.goal_x_velocity, self.goal_y_velocity, self.goal_yaw_velocity])
        observation[self.obs_idx.GOAL_VELOCITIES] = goal_velocity

        projected_gravity_vector = self.orientation_quat_inv.apply(np.array([0.0, 0.0, -1.0]))
        observation[self.obs_idx.PROJECTED_GRAVITY] = projected_gravity_vector

        if not self.action_space_mode == "default":
            observation[self.obs_idx.CPG] = self.cpg

        # observation[self.obs_idx.HEIGHT] = self.terrain_function.get_height_samples()

        # Normalize and clip
        for i, joint_range in enumerate(self.joint_order):
            observation[joint_range[0]] /= 3.14
            observation[joint_range[1]] /= self.max_joint_velocities[i]
            observation[joint_range[2]] /= 3.14
        # for i, foot_range in enumerate(self.feet_order):
        #     observation[foot_range[1]] = min(max(observation[foot_range[1]], 0.0), 5.0)
        observation[self.obs_idx.TRUNK_ANGULAR_VELOCITIES] /= 10.0
        return observation



    def get_reward_and_info(self, done):
        info = {"t": self.episode_step}
        reward, info = self.reward_function.reward_and_info(info, done)
        info = self.terrain_function.info(info)

        return reward, info


    def close(self):
        if self.mode == "test":
            pygame.quit()

    # time.sleep is not accurate under macos
    def precise_sleeping(self, waiting_time):
        i = 0
        now = time.perf_counter()
        end = now + waiting_time
        while now < end:
            i += 1
            now = time.perf_counter()
            time.sleep(0.0001)
