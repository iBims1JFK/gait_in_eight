import numpy as np
from scipy.spatial.transform import Rotation as R

class InverseKinematics:
    def __init__(self, env):
        self.env = env

        self.l1 = 0.0525   # Offset from J1 to J2 along [J1]X axis
        self.l2 = 0.18     # Length of femurr
        self.l3 = 0.177    # Length of tibia
        self.l4 = 0    # Offset from J1 to foot along [J1]Y axis

        self.l5 = 0.021     # Offset from J2 to foot along X 
        self.l6 = 0.175     # Offset foot height

        self.bodyLen = 0.367   #distance from front motor to rear motor
        self.bodyWidth = 0.1   #distance from left motor to right motor        // Leg lengths


        bodyToLegRotationEuler = [
            np.array([0.0, 0.0, 0.0]),
            np.array([np.pi, 0.0, 0.0]),
            np.array([np.pi, np.pi, 0.0]),
            np.array([0.0, np.pi, 0.0])
        ]

        self.bodyToLegRotationQuat = [R.from_euler('xyz', euler).as_quat() for euler in bodyToLegRotationEuler]
        self.bodyToLegRotationRotMat = [R.from_euler('xyz', euler).as_matrix() for euler in bodyToLegRotationEuler]

        self.bodyToLegTranslation = [
            np.array([ self.bodyLen / 2.0, -self.bodyWidth / 2.0,  0.0]),
            np.array([ self.bodyLen / 2.0,  self.bodyWidth / 2.0,  0.0]),
            np.array([-self.bodyLen / 2.0,  self.bodyWidth / 2.0,  0.0]),
            np.array([-self.bodyLen / 2.0, -self.bodyWidth / 2.0,  0.0])
        ]

        # order in xml rl, rr, fr, fl
        # order in ik fr, fl, rl, rr


    def process_action(self, action, offset):
        # offsets for a default leg position
        base = np.array([0.05, 0.02, -0.25])
        target_positions = np.zeros(12)
        for i in range(4):
            target = base.copy()

            # spread legs a bit
            if i in [0, 3]:
                target[1] *= -1
            target[2] += offset[i % 2]
            if i > 1:
                target[0] += -0.02
            index = (i + 2) % 4
            index = i
            target_positions[3*index:3*index+3] = self.calculate_ik_from_body(target, i) + action[3*i:3*i+3]
        
        return target_positions


    def calculate_ik_from_leg_origin(self, foot_pos_cart, leg_no, knee_backwards):
        if np.linalg.norm(foot_pos_cart)**2 > 0.335**2:
            foot_pos_cart = foot_pos_cart / np.linalg.norm(foot_pos_cart) * 0.335

        x, y, z = foot_pos_cart
        if leg_no in [1, 2]:
            y = -y

        l1 = self.l1
        l2 = self.l2
        l3 = self.l3
        l4 = self.l4

        if x == 0.0: x = 0.0001
        if y == 0.0: y = 0.0001
        if z == 0.0: z = 0.0001

        l_zy = np.sqrt(y**2 + z**2)
        x = x - l1  # Theta 1 - hip - computation
        t1_a = np.arcsin(l4 / l_zy)
        t1_b = np.arcsin(y / l_zy)
        theta1 = t1_a + t1_b

        # Theta 3 - knee - computation
        l_t1t2 = np.sqrt(l_zy**2 - l4**2)
        l_t1t2 = np.sqrt(l_t1t2**2 + x**2)  # Z compensation from X displacement
        theta3 = np.arccos((l_t1t2**2 - l2**2 - l3**2) / (-2.0 * l2 * l3))

        # Theta 2 - thigh
        t2_a = np.arcsin(x / l_t1t2)
        t2_b = np.arccos((l3**2 - l2**2 - l_t1t2**2) / (-2.0 * l2 * l_t1t2))

        if knee_backwards:
            theta2 = np.pi / 2 + t2_a - t2_b
            theta3 = -theta3
        else:
            theta2 = np.pi / 2 + t2_a + t2_b

        if leg_no in [1, 2]:
            theta1 = -theta1
            theta2 = -theta2
            theta3 = -theta3

        if leg_no in [2, 3]:
            theta1 = -theta1

        if theta1 > np.pi:
            theta1 = theta1 - 2 * np.pi
        if theta2 > np.pi:
            theta2 = theta2 - 2 * np.pi
        if theta3 > np.pi:
            theta3 = theta3 - 2 * np.pi

        if np.isnan(theta1): theta1 = 0.0
        if np.isnan(theta2): theta2 = 0.0
        if np.isnan(theta3): theta3 = 0.0

        return np.array([theta1, theta2, theta3])
    
    def calculate_ik_from_body(self, foot_pos_cart, leg_num):
        return self.calculate_ik_from_leg_origin(foot_pos_cart, leg_num, knee_backwards=True)

