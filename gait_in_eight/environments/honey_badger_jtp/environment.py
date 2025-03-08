from pathlib import Path
import psutil
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
from dm_control import mjcf
import pygame

from gait_in_eight.environments import observation_indices_wo_feet as obs_idx

from gait_in_eight.environments.honey_badger_jtp.viewer import MujocoViewer
from gait_in_eight.environments.honey_badger_jtp.control_functions.handler import get_control_function
from gait_in_eight.environments.honey_badger_jtp.command_functions.handler import get_command_function
from gait_in_eight.environments.honey_badger_jtp.sampling_functions.handler import get_sampling_function
from gait_in_eight.environments.honey_badger_jtp.initial_state_functions.handler import get_initial_state_function
from gait_in_eight.environments.honey_badger_jtp.reward_functions.handler import get_reward_function
from gait_in_eight.environments.honey_badger_jtp.termination_functions.handler import get_termination_function
from gait_in_eight.environments.honey_badger_jtp.domain_randomization.mujoco_model_functions.handler import get_domain_randomization_mujoco_model_function
from gait_in_eight.environments.honey_badger_jtp.domain_randomization.control_functions.handler import get_domain_randomization_control_function
from gait_in_eight.environments.honey_badger_jtp.domain_randomization.perturbation_functions.handler import get_domain_randomization_perturbation_function
from gait_in_eight.environments.honey_badger_jtp.observation_noise_functions.handler import get_observation_noise_function
from gait_in_eight.environments.honey_badger_jtp.terrain_functions.handler import get_terrain_function
from gait_in_eight.environments.honey_badger_jtp.trajectory_smoothing_functions.handler import get_trajectory_smoothing_function
from gait_in_eight.environments.honey_badger_cpg.evaluation_functions.handler import get_evaluation_function


class HoneyBadger(gym.Env):
    def __init__(self, seed, render,
                 mode,
                 control_type, control_frequency_hz, command_type, target_velocity, command_sampling_type, initial_state_type,
                 reward_type, termination_type,
                 domain_randomization_sampling_type,
                 domain_randomization_mujoco_model_type,
                 domain_randomization_control_type,
                 domain_randomization_perturbation_type, domain_randomization_perturbation_sampling_type,
                 observation_noise_type, terrain_type,
                 trajectory_smoothing_type,
                 trajectory_smoothing_history_length,
                 add_goal_arrow, timestep, episode_length_in_seconds, total_nr_envs, kp, kd,
                 cpu_id=None):
        
        if cpu_id is not None:
            p = psutil.Process()
            p.cpu_affinity([cpu_id,])

        self.kp = kp
        self.kd = kd
        self.trajectory_smoothing_history_length = trajectory_smoothing_history_length
        self.seed = seed
        self.mode = mode
        self.add_goal_arrow = add_goal_arrow
        self.total_nr_envs = total_nr_envs
        self.eval = False
        self.eval_at_last_setup = self.eval
        self.np_rng = np.random.default_rng(self.seed)
        self.target_velocity = target_velocity
        self.control_frequency_hz = control_frequency_hz
        self.nominal_joint_positions = np.array([
            -0.1, -0.6, 1.2,
            0.1, 0.6, -1.2,
            -0.1, 0.6, -1.2,
            0.1, -0.6, 1.2
        ])
        self.max_joint_velocities = np.array([
            25.0, 25.0, 25.0,
            25.0, 25.0, 25.0,
            25.0, 25.0, 25.0,
            25.0, 25.0, 25.0
        ])
        self.power_limit_watt = 1500
        self.initial_drop_height = 0.316

        self.total_timesteps = 0
        self.goal_x_velocity = 0
        self.goal_y_velocity = 0
        self.goal_yaw_velocity = 0

        self.simulation_step = 0
        self.terminated = False

        self.should_eval_feet_touchdown = True
        self.feet_touchdown_evaluation = [[], [], [], []]

        self.trunc_velocity_evaluation = []

        if mode == "test":
            initial_state_type = "default"
            domain_randomization_sampling_type = "none"
            domain_randomization_perturbation_sampling_type = "none"
            observation_noise_type = "none"

        self.timestep = timestep
        self.control_function = get_control_function(control_type, self)
        self.control_frequency_hz = self.control_function.control_frequency_hz
        self.nr_substeps = int(round(1 / self.control_frequency_hz / timestep))
        self.nr_intermediate_steps = 1
        self.dt = timestep * self.nr_substeps * self.nr_intermediate_steps
        self.horizon = int(round(episode_length_in_seconds * self.control_frequency_hz))
        self.command_function = get_command_function(command_type, self)
        self.command_sampling_function = get_sampling_function(command_sampling_type, self)
        self.initial_state_function = get_initial_state_function(initial_state_type, self)
        self.reward_function = get_reward_function(reward_type, self)
        self.termination_function = get_termination_function(termination_type, self)
        self.domain_randomization_sampling_function = get_sampling_function(domain_randomization_sampling_type, self)
        self.domain_randomization_mujoco_model_function = get_domain_randomization_mujoco_model_function(domain_randomization_mujoco_model_type, self)
        self.domain_randomization_control_function = get_domain_randomization_control_function(domain_randomization_control_type, self)
        self.domain_randomization_perturbation_function = get_domain_randomization_perturbation_function(domain_randomization_perturbation_type, self)
        self.domain_randomization_perturbation_sampling_function = get_sampling_function(domain_randomization_perturbation_sampling_type, self)
        self.observation_noise_function = get_observation_noise_function(observation_noise_type, self)
        self.terrain_function = get_terrain_function(terrain_type, self)
        self.evaluation_function = get_evaluation_function("default", self)

        xml_file_name = self.terrain_function.xml_file_name
        xml_path = (Path(__file__).resolve().parent.parent / "data" / xml_file_name).as_posix()
        if self.add_goal_arrow:
            # Add goal arrow
            xml_handle = mjcf.from_path(xml_path)
            trunk = xml_handle.find("body", "trunk")
            trunk.add("body", name="dir_arrow", pos="0 0 0.15")
            dir_vec = xml_handle.find("body", "dir_arrow")
            dir_vec.add("site", name="dir_arrow_ball", type="sphere", size=".02", pos="-.1 0 0")
            dir_vec.add("site", name="dir_arrow", type="cylinder", size=".01", fromto="0 0 -.1 0 0 .1")
            self.model = mujoco.MjModel.from_xml_string(xml=xml_handle.to_xml_string(), assets=xml_handle.get_assets())
        else:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self.trajectory_smoothing_function = get_trajectory_smoothing_function(trajectory_smoothing_type, self)

        collision_groups = [("floor", ["floor"]),
                            ("feet", ["RL_foot", "RR_foot", "FR_foot", "FL_foot"]),
                            ("RL_foot", ["RL_foot"]), ("RR_foot", ["RR_foot"]), ("FR_foot", ["FR_foot"]), ("FL_foot", ["FL_foot"]),
                            ("RL_calf", ["RL_calf"]), ("RR_calf", ["RR_calf"]), ("FR_calf", ["FR_calf"]), ("FL_calf", ["FL_calf"]),
                            ("trunk", ["trunk_1"]),
                            ("trunk_1", ["trunk_1"])]
        self.collision_groups = {}
        if collision_groups is not None:
            for name, geom_names in collision_groups:
                self.collision_groups[name] = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name) for geom_name in geom_names}

        self.viewer = None if not render else MujocoViewer(self.model, self.dt)
        
        action_space_low = -np.asarray([0.2, 0.4, 0.4] * 4)
        action_space_high = np.asarray([0.2, 0.4, 0.4] * 4) 
        self.action_space = gym.spaces.Box(low=action_space_low, high=action_space_high, dtype=np.float32)

        self.joint_order = [
            obs_idx.QUADRUPED_BACK_LEFT_HIP, obs_idx.QUADRUPED_BACK_LEFT_THIGH, obs_idx.QUADRUPED_BACK_LEFT_CALF,
            obs_idx.QUADRUPED_BACK_RIGHT_HIP, obs_idx.QUADRUPED_BACK_RIGHT_THIGH, obs_idx.QUADRUPED_BACK_RIGHT_CALF,
            obs_idx.QUADRUPED_FRONT_RIGHT_HIP, obs_idx.QUADRUPED_FRONT_RIGHT_THIGH, obs_idx.QUADRUPED_FRONT_RIGHT_CALF,
            obs_idx.QUADRUPED_FRONT_LEFT_HIP, obs_idx.QUADRUPED_FRONT_LEFT_THIGH, obs_idx.QUADRUPED_FRONT_LEFT_CALF
        ]

        self.foot_names = ["RL_foot", "RR_foot", "FR_foot", "FL_foot"]
        self.joint_names = [
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        ]

        qpos, qvel = get_initial_state_function("default", self).setup()
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

        self.update_orientation_attributes()
        
        self.observation_space = self.get_observation_space()

        self.current_action = np.zeros(self.model.nu)
        self.initial_observation = self.get_initial_observation()
        self.next_observation_history = np.zeros((5, self.initial_observation.shape[0]))

        self.reward_function.init()
        self.domain_randomization_mujoco_model_function.init()
        self.observation_noise_function.init()
        self.trajectory_smoothing_function.init()

        if self.mode == "test":
            pygame.init()
            pygame.joystick.init()
            self.joystick_present = False
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                self.joystick_present = True


    def reset(self, seed=None):
        self.episode_step = 0
        self.current_action = np.zeros(self.model.nu)
        self.last_action = np.zeros(self.model.nu)
        self.current_torques = np.zeros(self.model.nu)
        self.trajectory_smoothing_function.init()

        self.termination_function.setup()
        self.terrain_function.sample()
        self.command_function.setup()
        self.reward_function.setup()
        self.handle_domain_randomization(function="setup")
        self.evaluation_function.setup()

        qpos, qvel = self.initial_state_function.setup()

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

        self.update_orientation_attributes()

        if self.viewer:
            self.viewer.render(self.data)
        if self.mode == "test":
            if self.should_eval_feet_touchdown:
                save_feet_touchdown = np.asarray(self.feet_touchdown_evaluation)
                if save_feet_touchdown.size > 0:  # Ensure there is data to save
                    np.save("feet_touchdown_evaluation.npy", save_feet_touchdown)
                self.feet_touchdown_evaluation = [[], [], [], []]
                save_trunc_velocity = np.asarray(self.trunc_velocity_evaluation)
                if save_trunc_velocity.size > 0:
                    np.save("trunc_velocity_evaluation.npy", save_trunc_velocity)
                self.trunc_velocity_evaluation = []


        return self.get_observation(), {}


    def step(self, action):
        explicit_commands = False
        if self.mode == "test":
            if self.joystick_present:
                pygame.event.pump()
                self.goal_x_velocity = -self.joystick.get_axis(1)
                self.goal_y_velocity = -self.joystick.get_axis(0)
                self.goal_yaw_velocity = -self.joystick.get_axis(3)
            elif Path("commands.txt").is_file():
                with open("commands.txt", "r") as f:
                    commands = f.readlines()
                if len(commands) == 3:
                    self.goal_x_velocity = float(commands[0])
                    self.goal_y_velocity = float(commands[1])
                    self.goal_yaw_velocity = float(commands[2])
                    explicit_commands = True
            if self.should_eval_feet_touchdown:
                self.feet_touchdown_evaluation[0].append(1 if self.check_collision("floor", "FL_foot") else 0)
                self.feet_touchdown_evaluation[1].append(1 if self.check_collision("floor", "FR_foot") else 0)
                self.feet_touchdown_evaluation[2].append(1 if self.check_collision("floor", "RL_foot") else 0)
                self.feet_touchdown_evaluation[3].append(1 if self.check_collision("floor", "RR_foot") else 0)

        if not explicit_commands:
            should_sample_commands = self.command_sampling_function.step()
            if should_sample_commands or self.total_timesteps == 0:
                if self.eval:
                    self.goal_x_velocity, self.goal_y_velocity, self.goal_yaw_velocity = self.evaluation_function.get_next_command()
                else:
                    self.goal_x_velocity, self.goal_y_velocity, self.goal_yaw_velocity = self.command_function.get_next_command()


        action = self.trajectory_smoothing_function.step(action)
        for i in range(self.nr_substeps):
            torques = self.control_function.process_action(action)
            self.data.ctrl = torques
            mujoco.mj_step(self.model, self.data, 1)
            self.data.qvel[6:] = np.clip(
                self.data.qvel[6:],
                -self.max_joint_velocities,
                self.max_joint_velocities,
            )

            if self.mode == "test" and self.should_eval_feet_touchdown:
                global_linear_velocity = self.data.qvel[:3]
                local_linear_velocity = self.orientation_quat_inv.apply(global_linear_velocity)
                self.trunc_velocity_evaluation.append(local_linear_velocity[0])

        self.simulation_step += self.nr_substeps

        self.update_orientation_attributes()

        if self.add_goal_arrow:
            trunk_rotation = self.orientation_euler[2]
            desired_angle = trunk_rotation + np.arctan2(self.goal_y_velocity, self.goal_x_velocity)
            rot_mat = R.from_euler('xyz', (np.array([np.pi/2, 0, np.pi/2 + desired_angle]))).as_matrix()
            self.data.site("dir_arrow").xmat = rot_mat.reshape((9,))
            magnitude = np.sqrt(np.sum(np.square([self.goal_x_velocity, self.goal_y_velocity])))
            self.model.site_size[1, 1] = magnitude * 0.1
            arrow_offset = -(0.1 - (magnitude * 0.1))
            self.data.site("dir_arrow").xpos += [arrow_offset * np.sin(np.pi/2 + desired_angle), -arrow_offset * np.cos(np.pi/2 + desired_angle), 0]
            self.data.site("dir_arrow_ball").xpos = self.data.body("dir_arrow").xpos + [-0.1 * np.sin(np.pi/2 + desired_angle), 0.1 * np.cos(np.pi/2 + desired_angle), 0]

        if self.viewer:
            self.viewer.render(self.data)
        
        self.current_action = action.copy()
        self.current_torques = torques

        self.handle_domain_randomization(function="step")

        self.next_observation = self.get_observation()
        self.next_observation_history = np.roll(self.next_observation_history, 1, axis=0)
        self.next_observation_history[0] = self.next_observation
        terminated = self.termination_function.should_terminate(self.next_observation)
        self.terminated = terminated
        truncated = self.episode_step + 1 >= self.horizon
        done = terminated | truncated
        reward, r_info = self.get_reward_and_info(done)
        info = {**r_info}
        info["simulation_step"] = self.simulation_step

        self.reward_function.step(action)
        self.command_function.step(self.next_observation, reward, done, info)
        if self.eval:
            self.evaluation_function.step(self.next_observation, reward, done, info)
        self.initial_state_function.step(self.next_observation, reward, done, info)
        self.terrain_function.step(self.next_observation, reward, done, info)
        
        self.last_action = action.copy()
        self.episode_step += 1
        if not self.eval:
            self.total_timesteps += 1

        return self.next_observation, reward, terminated, truncated, info


    def update_orientation_attributes(self):
        self.orientation_quat = R.from_quat([self.data.qpos[4], self.data.qpos[5], self.data.qpos[6], self.data.qpos[3]])
        self.orientation_euler = self.orientation_quat.as_euler("xyz")
        self.orientation_quat_inv = self.orientation_quat.inv()


    def handle_domain_randomization(self, function="setup"):
        if function == "setup":
            if self.eval_at_last_setup != self.eval:
                self.should_randomize_domain = True
                self.should_randomize_domain_perturbation = True
                self.eval_at_last_setup = self.eval
            else:
                self.should_randomize_domain = self.domain_randomization_sampling_function.setup()
                self.should_randomize_domain_perturbation = self.domain_randomization_perturbation_sampling_function.setup()
        elif function == "step":
            self.should_randomize_domain = self.domain_randomization_sampling_function.step()
            self.should_randomize_domain_perturbation = self.domain_randomization_perturbation_sampling_function.step()
        if self.should_randomize_domain:
            self.domain_randomization_control_function.sample()
            self.domain_randomization_mujoco_model_function.sample()
            self.reward_function.init()
        if self.should_randomize_domain_perturbation:
            self.domain_randomization_perturbation_function.sample()


    def get_observation_space(self):
        space_low = np.array([-np.inf] * obs_idx.OBSERVATION_SIZE, dtype=np.float32)
        space_high = np.array([np.inf] * obs_idx.OBSERVATION_SIZE, dtype=np.float32)

        return gym.spaces.Box(low=space_low, high=space_high, shape=space_low.shape, dtype=np.float32)


    def get_initial_observation(self):
        return np.zeros(obs_idx.OBSERVATION_SIZE, dtype=np.float32)
    

    def get_observation(self):
        observation = self.initial_observation.copy()

        # Dynamic observations
        for i, joint_range in enumerate(self.joint_order):
            observation[joint_range[0]] = self.data.qpos[i+7] - self.nominal_joint_positions[i]
            observation[joint_range[1]] = self.data.qvel[i+6]
            observation[joint_range[2]] = self.current_action[i]

        # IMU accelerations
        trunk_linear_acceleration = self.orientation_quat_inv.apply(self.data.qacc[:3])
        observation[obs_idx.TRUNK_LINEAR_ACCELERATIONS] = trunk_linear_acceleration

        trunk_angular_acceleration = self.data.qacc[3:6]
        observation[obs_idx.TRUNK_ANGULAR_ACCELERATIONS] = trunk_angular_acceleration
        

        # General observations
        trunk_linear_velocity = self.orientation_quat_inv.apply(self.data.qvel[:3])
        observation[obs_idx.TRUNK_LINEAR_VELOCITIES] = trunk_linear_velocity

        trunk_angular_velocity = self.data.qvel[3:6]
        observation[obs_idx.TRUNK_ANGULAR_VELOCITIES] = trunk_angular_velocity

        goal_velocity = np.array([self.goal_x_velocity, self.goal_y_velocity, self.goal_yaw_velocity])
        observation[obs_idx.GOAL_VELOCITIES] = goal_velocity

        projected_gravity_vector = self.orientation_quat_inv.apply(np.array([0.0, 0.0, -1.0]))
        observation[obs_idx.PROJECTED_GRAVITY] = projected_gravity_vector
    
        # Add noise
        observation = self.observation_noise_function.modify_observation(observation)

        # Normalize and clip
        for i, joint_range in enumerate(self.joint_order):
            observation[joint_range[0]] /= 3.14
            observation[joint_range[1]] /= self.max_joint_velocities[i]
            observation[joint_range[2]] /= 3.14
        observation[obs_idx.TRUNK_ANGULAR_VELOCITIES] /= 10.0
        return observation


    def get_reward_and_info(self, done):
        info = {"t": self.episode_step}
        reward, info = self.reward_function.reward_and_info(info, done)
        info = self.terrain_function.info(info)

        return reward, info


    def close(self):
        if self.viewer:
            self.viewer.close()
        if self.mode == "test":
            pygame.quit()


    def check_collision(self, groups1, groups2):
        if isinstance(groups1, list):
            ids1 = [self.collision_groups[group] for group in groups1]
            ids1 = set().union(*ids1)
        else:
            ids1 = self.collision_groups[groups1]
        
        if isinstance(groups2, list):
            ids2 = [self.collision_groups[group] for group in groups2]
            ids2 = set().union(*ids2)
        else:
            ids2 = self.collision_groups[groups2]

        for coni in range(0, self.data.ncon):
            con = self.data.contact[coni]

            collision = con.geom1 in ids1 and con.geom2 in ids2
            collision_trans = con.geom1 in ids2 and con.geom2 in ids1

            if collision or collision_trans:
                return True

        return False
    

    def check_any_collision(self, groups):
        if isinstance(groups, list):
            ids = [self.collision_groups[group] for group in groups]
            ids = set().union(*ids)
        else:
            ids = self.collision_groups[groups]

        for con_i in range(0, self.data.ncon):
            con = self.data.contact[con_i]
            if con.geom1 in ids or con.geom2 in ids:
                return True
        
        return False


    def check_any_collision_for_all(self, groups):
        ids = [self.collision_groups[group] for group in groups]
        ids = set().union(*ids)

        any_collision = {idx: False for idx in ids}

        for con_i in range(0, self.data.ncon):
            con = self.data.contact[con_i]
            if con.geom1 in ids:
                any_collision[con.geom1] = True
                ids.remove(con.geom1)
            if con.geom2 in ids:
                any_collision[con.geom2] = True
                ids.remove(con.geom2)
        
        return any_collision

    def set_eval_mode(self, eval):
        self.eval = eval