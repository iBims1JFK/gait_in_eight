import numpy as np
import rclpy
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from gait_in_eight.environments.honey_badger_ros.ros_connection.remote import Remote
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from hb40_commons.msg import JointCommand, BridgeData, RobotState
import rclpy.qos
from rclpy.qos import QoSProfile
import time
from gait_in_eight.environments.honey_badger_ros.ros_connection.kalman_filter import AccelerationEKF    
from gait_in_eight.environments.honey_badger_ros.ros_connection.kalman_filter import LinearVelocityKF as EKF    
from filterpy.kalman import KalmanFilter
from scipy import integrate
from gait_in_eight.environments.honey_badger_ros.ros_connection.cpg import CPG
import threading
import cv2
import signal
import sys
from OneEuroFilter import OneEuroFilter

joint_cmd_topic = "/hb40/joint_command"
bridge_data_topic = "/hb40/bridge_data"
velocity_data_topic = "/hb40/robot_state"
twist_velocity_data_topic = "/hb40/trunk_vel"

kp=60.0
kd=3.0
velocity_estimation_type = "onboard" # "imu", "onboard", "ground_truth", "april"
track_velocity_with_april = False

if velocity_estimation_type == "april" or track_velocity_with_april:
    from gait_in_eight.environments.honey_badger_ros.ros_connection.apriltag_estimation import ApriltagEstimation
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge


class RosConnectionNode(Node):

    def __init__(self, joint_order :list, nominal_position :np.ndarray, env, remote :Remote):
        super().__init__('ros_connection_node')
        print("Vel estimator started?")

        self.env = env
        self.acceleration_history = np.zeros((800 * 100,3))
        self.time_history = np.zeros(800 * 100)
        self.saving = False
        
        self.standing_up_progress = 0.0
        self.remote = remote
        self.i = 0
        self.joint_order = joint_order
        self.nominal_position = nominal_position
        self.last_imu_msg_received = None
        self.measurement_var = 0.00134 ** 2
        self.process_var = 0.0001
        self.base_orientation = np.eye(3)
        self.update_base_orientation = True
        self.filter = EKF(0.07)
        self.acc_filter = AccelerationEKF(1/416.67, self.process_var, self.measurement_var)
        self.filter_time = time.perf_counter()
        
        if not self.env.action_space_mode == "default":
            self.cpg = CPG(env, 1.75, self.env.action_space_mode)

        qos_joint_command = QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            depth=rclpy.qos.HistoryPolicy.KEEP_LAST,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE
        )

        self.joint_command_pub_ = self.create_publisher(JointCommand, joint_cmd_topic, qos_joint_command)

        qos_profil = QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            depth=rclpy.qos.HistoryPolicy.UNKNOWN,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            liveliness=rclpy.qos.LivelinessPolicy.AUTOMATIC
        )
        self.bridge_data_sub_ = self.create_subscription(BridgeData, bridge_data_topic, self.bridge_data_callback, qos_profil)
        if velocity_estimation_type == "onboard":
            self.velocity_data_sub_ = self.create_subscription(RobotState, velocity_data_topic, self.velocity_data_callback, qos_profil)
        # ground truth from simulation
        elif velocity_estimation_type == "ground_truth":
            self.velocity_data_sub_ = self.create_subscription(Twist, twist_velocity_data_topic, self.twist_velocity_data_callback, qos_profil)
        
        self.debug_pub_ = self.create_publisher(Twist, "/on_robot/debug", 10)
        self.velocity_pub_is_live = False

        if velocity_estimation_type == "april":
            self.image_publisher_ = self.create_publisher(Image, '/on_robot/image', 10)
            self.image_timer = self.create_timer(0.2, self.image_timer_callback)
            self.br = CvBridge()

        self.hz = 1000
        self.joint_msg = self.collapse_msg()


        timer_callback_group = rclpy.callback_groups.ReentrantCallbackGroup()
        self.timer = self.create_timer(1/self.hz, self.timer_callback, callback_group=timer_callback_group)
        self.state_timer = self.create_timer(1 / self.hz, self.state_timer_callback)
        if not self.env.action_space_mode == "default":
            cpg_timer_callback_group = rclpy.callback_groups.ReentrantCallbackGroup()
            self.cpg_timer = self.create_timer(1 / 40, self.cpg_timer_callback, callback_group=cpg_timer_callback_group)
            self.cpg_time = time.perf_counter()

        self.qpos = np.zeros(7 + 12)
        self.qpos[3] = 1
        self.qvel = np.zeros(6 + 12)
        self.qacc = np.zeros(6 + 12)
        self.ctrl = np.zeros(12)

        self.imu_calibration_data = []
        self.imu_correction = 1.0

        self.april_qvel = None

        # apriltag estimation thread
        signal.signal(signal.SIGINT, self.signal_handler)
        if velocity_estimation_type == "april" or track_velocity_with_april:
            self.apriltag_estimation = ApriltagEstimation(self.filter, render=False)
            thread = threading.Thread(target=self.apriltag_estimation.run, daemon=True)
            thread.start()


    def send_joint_command(self, action :np.ndarray):
        if not self.remote.get_state() == Remote.LEARN:
            return
        if not self.velocity_pub_is_live:
            print("no velocity data")
        msg = self.default_msg()
        msg.t_pos = [*(action.tolist()), 0.0]
        self.joint_msg = msg

    def bridge_data_callback(self, msg: BridgeData):
        if not self.velocity_pub_is_live:
            print("no velocity data")
        qacc = np.zeros(6 + 12)
        qacc[0] = msg.linear_acceleration.x
        qacc[1] = msg.linear_acceleration.y
        qacc[2] = msg.linear_acceleration.z
        qacc *= self.imu_correction
        self.qacc = qacc

        if self.update_base_orientation:
            self.base_orientation = R.from_quat([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]).as_matrix()
            self.update_base_orientation = False

        qpos = np.zeros(7 + 12)
        qpos[3] = msg.orientation.w
        qpos[4] = msg.orientation.x
        qpos[5] = msg.orientation.y
        qpos[6] = msg.orientation.z
        qvel = np.zeros(6 + 12)

        if velocity_estimation_type == "imu":
            qvel[:3] = self.calc_velocity_from_imu(msg)
        if velocity_estimation_type == "april" or track_velocity_with_april:
            self.april_qvel = self.calc_velocity_from_imu_and_april(msg)
            if velocity_estimation_type == "april":
                qvel[:3] = self.april_qvel

        qvel[3] = msg.angular_velocity.x
        qvel[4] = msg.angular_velocity.y
        qvel[5] = msg.angular_velocity.z
        
        for i, joint in enumerate(self.joint_order):
            # joint_index = msg.joint_name.index(joint)
            joint_index = i
            qpos[i + 7] = msg.joint_position[joint_index]
            qvel[i + 6] = msg.joint_velocity[joint_index]
        
        self.saving = True
        self.qpos = qpos
        if velocity_estimation_type == "onboard":
            self.qvel[3:] = qvel[3:]
        else:
            self.qvel = qvel
        self.saving = False

        self.ctrl = np.asarray(msg.joint_effort)
        joint_angle_safe = self.joint_angle_check(qpos[7:])
        orientation_safe = self.orientation_check(self.qpos)
        if not joint_angle_safe or not orientation_safe:
            self.remote.set_state(Remote.COLLAPSE)
    
    def calc_velocity_from_imu_and_april(self, msg: BridgeData):
        self.velocity_pub_is_live = True
        
        local_r_acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        local_to_global_imu = R.from_quat([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]).as_matrix()
        self.apriltag_estimation.set_imu_orientation(local_to_global_imu)
        skip = False

        global_imu_acc = local_to_global_imu @ local_r_acc
        global_imu_acc *= self.imu_correction
        global_imu_acc[2] -= 9.81


        try:
            local_r_to_global_r = self.apriltag_estimation.local_r_to_global_r
            skip = False
        except:
            skip = True
        if not skip:
            current_time = time.perf_counter()
            diff = current_time - self.filter_time
            self.filter_time = current_time

            if self.apriltag_estimation.tracking_mode == "april_only":
                state = self.filter.get_state()
                global_r_coord = np.array([state[0], state[2], state[4]])
                global_r_vel = np.array([state[1], state[3], state[5]])
                local_r_vel = self.apriltag_estimation.global_c_vel_to_local_r @ self.apriltag_estimation.camera_to_robot.T @ global_r_vel
            elif self.apriltag_estimation.tracking_mode == "acc_and_april_imu":
                self.filter.predict(global_imu_acc, 1/416.67)
                state = self.filter.get_state()
                global_imu_coord = np.array([state[0], state[2], state[4]])
                global_imu_vel = np.array([state[1], state[3], state[5]])
                local_r_vel = self.apriltag_estimation.local_to_global_imu @ global_imu_vel
            elif self.apriltag_estimation.tracking_mode == "acc_and_april_camera":
                local_r_acc = local_to_global_imu.T @ global_imu_acc
                global_r_acc = self.apriltag_estimation.local_r_to_global_r @ local_r_acc
                # left hand coordinate system
                global_r_acc *= np.array([1, -1, 1])
                self.filter.predict(global_r_acc, 1/416.67)
                state = self.filter.get_state()
                global_r_coord = np.array([state[0], state[2], state[4]])
                global_r_vel = np.array([state[1], state[3], state[5]])
                local_r_vel = self.apriltag_estimation.global_c_vel_to_local_r @ self.apriltag_estimation.camera_to_robot.T @ global_r_vel

            qvel = np.zeros(3)
            qvel = local_r_vel
            return qvel
            
    
    def velocity_data_callback(self, msg: RobotState):
        self.velocity_pub_is_live = True
        global_vel = np.array([msg.world_vel.linear.x, msg.world_vel.linear.y, msg.world_vel.linear.z])
        local_vel = np.array([msg.body_vel.linear.x, msg.body_vel.linear.y, msg.body_vel.linear.z])
        qvel = self.qvel
        qvel[:3] = local_vel
        debug_msg = Twist()
        debug_msg.linear.x = global_vel[0]
        debug_msg.linear.y = global_vel[1]
        debug_msg.linear.z = global_vel[2]
        self.debug_pub_.publish(debug_msg)
    
    def calc_velocity_from_imu(self, msg: BridgeData):
        current_time = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
        self.velocity_pub_is_live = True
        if self.last_imu_msg_received is None:
            self.last_imu_msg_received = current_time
            local_acceleration = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])        
            global_orientation = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
            quat = R.from_quat(global_orientation)
            global_acceleration = quat.apply(local_acceleration)
            self.last_acceleration = global_acceleration
            return self.qvel[:3]
        diff = (current_time - self.last_imu_msg_received) / 1e9
        diff = 1/416.667    
        local_acceleration = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        global_orientation = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        quat = R.from_quat(global_orientation)
        global_acceleration = quat.apply(local_acceleration)
        global_acceleration[2] -= 9.81
        
        self.acc_filter.predict()
        self.acc_filter.update(global_acceleration)
        filtered_velocity = self.acc_filter.x[3:6]
        updated_acceleration = global_acceleration

        qvel = filtered_velocity
        self.last_imu_msg_received = current_time
        self.last_acceleration = updated_acceleration
        local_vel = quat.inv().apply(qvel)
        debug_msg = Twist()
        debug_msg.linear.x = local_vel[0]
        debug_msg.linear.y = local_vel[1]
        debug_msg.linear.z = local_vel[2]
        self.debug_pub_.publish(debug_msg)
        return local_vel

    # debug function for ground truth velocity emitted from simulation 
    def twist_velocity_data_callback(self, msg: Twist):
        self.velocity_pub_is_live = True
        quat = R.from_quat([self.qpos[4], self.qpos[5], self.qpos[6], self.qpos[3]])
        local_vel = quat.inv().apply(np.array([msg.linear.x, msg.linear.y, msg.linear.z]))
        world_vel = np.array([msg.linear.x, msg.linear.y, msg.linear.z])
        qvel = np.zeros(3)
        qvel[0] = local_vel[0]
        qvel[1] = local_vel[1]
        qvel[2] = local_vel[2]
        self.qvel[:3] = qvel
    
    def get_data(self):
        return self.qpos, self.qvel, self.qacc, self.ctrl, self.april_qvel

    def standing(self):
        msg = self.default_msg()
        msg.t_pos = [*(self.nominal_position), 0.0]
        msg.kp = [*([25.0] * 12), 0.0]
        self.joint_msg = msg

    def standing_up(self):
        self.pose(self.nominal_position)

    def pose(self, desired):
        sec = 1.5
        target = 30
        steps = self.hz * sec
        msg = self.default_msg()
        msg.kp = [*([self.standing_up_progress] * 12), 0.0]
        msg.t_pos = [*(desired), 0.0]
        msg.name = [*self.joint_order, "sp_j0"]
        self.joint_msg = msg

        self.standing_up_progress = self.standing_up_progress + target / steps
        self.standing_up_progress = min(target, self.standing_up_progress)
        if self.standing_up_progress == target:
            self.remote.set_state(Remote.STANDING)
            self.standing_up_progress = 0.0
            
            # nedd to update base orientation of magnetometer and tag estimation
            self.update_base_orientation = True
            if velocity_estimation_type == "april":
                self.apriltag_estimation.set_update_base_orientation()


    def robot_collapse(self):
        msg = self.collapse_msg()
        self.joint_msg = msg
    
    def default_msg(self):
        msg = JointCommand()
        msg.source_node = "ros_connection_node"
        msg.name = [*self.joint_order, "sp_j0"]
        msg.t_pos = [0.0] * 13
        msg.t_vel = [0.0] * 13
        msg.t_trq = [0.0] * 13
        msg.kp = [*([kp] * 12), 0.0]
        msg.kd = [*([kd] * 12), 0.0]

        return msg
    
    def collapse_msg(self):
        msg = JointCommand()
        msg.source_node = "ros_connection_node"
        msg.name = [*self.joint_order, "sp_j0"]
        msg.t_pos = [0.0] * 13
        msg.t_vel = [0.0] * 13
        msg.t_trq = [0.0] * 13
        msg.kp = [0.0] * 13
        msg.kd = [5.0] * 13

        return msg    
    
    def joint_angle_check(self, qpos):
        joint_limits_min = np.array([
	        -1.57, -1.57, -2.53, -1.57, -1.57, 0., -1.57, -1.57, 0., -1.57, -1.57, -2.53])
        joint_limits_max = np.array([
            1.57, 1.57, 0., 1.57, 1.57, 2.53, 1.57, 1.57, 2.53, 1.57, 1.57, 0.
        ])
        angle_offset = 0.08726

        return (joint_limits_min + angle_offset < qpos).any() or (qpos < joint_limits_max - angle_offset).any()
    
    def imu_calibration(self):
        if velocity_estimation_type == "imu":
            self.qvel[:3] = np.zeros(3)
            self.acc_filter.reset()
        if len(self.imu_calibration_data) < 100:
            self.imu_calibration_data.append(self.qacc[:3])
        else:
            data = np.asarray(self.imu_calibration_data)
            avg = np.mean(data, axis=0)
            norm = np.linalg.norm(avg)
            self.imu_correction = 9.81 / norm
            self.imu_calibration_data = []
            self.remote.set_state(Remote.LEARN)

    
    def orientation_check(self, qpos):
        rollmargin = 1.5
        pitchmargin = 1.5
        rpy = R.from_quat([qpos[4], qpos[5], qpos[6], qpos[3]]).as_euler("xyz")
        return np.abs(rpy[0]) < rollmargin and np.abs(rpy[1]) < pitchmargin

    def timer_callback(self):
        self.joint_command_pub_.publish(self.joint_msg)
    
    def image_timer_callback(self):
        if self.apriltag_estimation.image is not None:
            image = self.br.cv2_to_imgmsg(self.apriltag_estimation.image)
            self.image_publisher_.publish(image)

    def state_timer_callback(self):    
        if self.remote.get_state() == Remote.COLLAPSE or self.remote.get_state() == Remote.IDLE:
            self.robot_collapse()
        elif self.remote.get_state() == Remote.STANDING or self.remote.get_state() == Remote.PAUSE:
            self.standing()
        elif self.remote.get_state() == Remote.STANDING_UP:
            self.standing_up()
        elif self.remote.get_state() == Remote.IMU_CALIBRATION:
            self.imu_calibration()
            self.standing()

        if not self.remote.get_state() == Remote.STANDING_UP and not self.standing_up_progress == 0:
            self.standing_up_progress = 0.0
    
    def cpg_timer_callback(self):
        current_time = time.perf_counter()
        if not self.cpg_time is None and 1 / (current_time - self.cpg_time) < 39:
            print(f"Frequency: {1 / (current_time - self.cpg_time)}")
        self.cpg_time = current_time
        if self.remote.get_state() == Remote.LEARN:
            joint_target = self.cpg.step()
            self.send_joint_command(joint_target)
    
    def set_action(self, action):
        self.cpg.set_action(action)

    def signal_handler(self, sig, frame):
        if velocity_estimation_type == "april" or track_velocity_with_april:
            self.apriltag_estimation.shutdown()
        sys.exit(0)
