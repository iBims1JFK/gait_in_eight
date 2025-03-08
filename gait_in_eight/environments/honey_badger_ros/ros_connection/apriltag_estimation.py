# import the opencv library 
import cv2 
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from dt_apriltags import Detector
from filterpy.kalman import KalmanFilter
default_config = {
    'families': 'tagStandard41h12',
    'nthreads': 3,
    'quad_decimate': 1.5,
    'quad_sigma': 0.0,
    'refine_edges': 1,
    'decode_sharpening': 10.0,
    'max_hamming': 0,
    'debug': 0
}

camera_params = [1.39436579e+03, 1.34334001e+03, 9.36256161e+02, 4.98152894e+02]
camera_matrix = np.array([[camera_params[0], 0, camera_params[2]], [0, camera_params[1], camera_params[3]], [0, 0, 1]])
cube_mode = "cube"
if cube_mode == "cube":
    tag_size = 0.088
else:
    tag_size = 0.114

tracking_mode = "april_only"    
# tracking_mode = "acc_and_april_imu"
# tracking_mode = "acc_and_april_camera"
# tracking_mode = "acc_and_april_local"

STATES = {
    'INIT': "init",
    'RUNNING': "running",
}

camera_to_robot = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, -1, 0]   
])



def rotation_z(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])


class ApriltagEstimation:
    def __init__(self, ekf=None, render=False, detector_config=default_config):
        self.ekf = ekf
        self.render = render
        self.record = False
        self.detector = Detector(**detector_config)
        self.state = STATES['INIT']
        self.vel = {'current_vel': np.zeros(3), 'last_measurement': time.perf_counter(), 'last_pos': np.zeros(3)}
        self.vel = {'current_vel': 0, 'last_measurement': time.perf_counter(), 'last_pos': 0}
        self.camera_to_robot = camera_to_robot
        self.base_orientation = np.eye(3)
        self.update_base_orientation = True
        self.local_to_global_imu = np.eye(3)
        self.image = None
        self.tracking_mode = tracking_mode


    def get_tags(self, frame):
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # self.image = gray_scale
        dec_img = np.asarray(gray_scale)
        tags = self.detector.detect(dec_img, True, camera_params, tag_size)
        return tags
    
    def render_tags(self, frame, tags):
        origin = np.zeros(2)
        ot = np.zeros(3)
        for tag in tags:
            for idx in range(len(tag.corners)):
                point1 = tuple(tag.corners[idx-1].astype(int).tolist())
                point2 = tuple(tag.corners[idx].astype(int).tolist())
                cv2.line(frame, point1, point2, (0, 255, 0), 2)
            origin = tag.center
            ot = np.asarray(tag.pose_t)
            
        skip = False
        try:
            self.global_r_to_local_r
            skip = False
        except:
            print("undefined")
            skip = True
        
        if not skip:
            state = self.ekf.get_state()
            if tracking_mode == "april_only":
                global_r_coord = np.array([state[0], state[2], state[4]])
                global_r_vel = np.array([state[1], state[3], state[5]])

                # in robot orientation
                local_r_vel = self.global_c_vel_to_local_r @ self.camera_to_robot.T @ global_r_vel
                local_r_coord = self.global_c_vel_to_local_r @ self.camera_to_robot.T @ global_r_coord
            elif tracking_mode == "acc_and_april_imu":
                global_imu_coord = np.array([state[0], state[2], state[4]])
                global_imu_vel = np.array([state[1], state[3], state[5]])

                # in robot orientation
                local_r_coord = self.local_to_global_imu @ global_imu_coord
                global_r_coord = self.camera_to_robot @ self.global_c_vel_to_local_r.T @ local_r_coord
                local_r_vel = self.local_to_global_imu @ global_imu_vel
            elif tracking_mode == "acc_and_april_camera":
                global_r_coord = np.array([state[0], state[2], state[4]])
                global_r_vel = np.array([state[1], state[3], state[5]])

                # in robot orientation
                local_r_vel = self.global_c_vel_to_local_r @ self.camera_to_robot.T @ global_r_vel
                local_r_coord = self.global_c_vel_to_local_r @ self.camera_to_robot.T @ global_r_coord

            cv2.putText(frame, f"Estimated State: Position = ({global_r_coord[0]:.2f}, {global_r_coord[1]:.2f}, {global_r_coord[2]:.2f})", 
                org=(10, 70),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0, 0, 255),
                thickness=3)

            cv2.putText(frame, f"Velocity: {local_r_vel[0]:.2f}m/s {local_r_vel[1]:.2f}m/s {local_r_vel[2]:.2f}m/s",
                org=(10, 120),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0, 0, 255),
                thickness=3)



            points = np.array([[local_r_vel[0], 0, 0], [0, local_r_vel[1], 0], [0, 0, local_r_vel[2]]]).reshape(-1, 3)
            axisPoints, _ = cv2.projectPoints(points, self.global_c_vel_to_local_r.T, self.camera_to_robot.T @ global_r_coord, camera_matrix, (0,0,0,0))

            originAxisPoints, _ = cv2.projectPoints(local_r_coord, self.global_c_vel_to_local_r.T, self.camera_to_robot.T @ global_r_coord, camera_matrix, (0,0,0,0))
            origin = originAxisPoints.flatten()

            cv2.line(frame, tuple(origin.astype(int)), tuple(axisPoints[0].ravel().astype(int)), (255,0,0), 3)
            cv2.line(frame, tuple(origin.astype(int)), tuple(axisPoints[1].ravel().astype(int)), (0,255,0), 3)
            cv2.line(frame, tuple(origin.astype(int)), tuple(axisPoints[2].ravel().astype(int)), (0,0,255), 3)

            cv2.putText(frame, f"X", tuple(axisPoints[0].ravel().astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.putText(frame, f"Y", tuple(axisPoints[1].ravel().astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f"Z", tuple(axisPoints[2].ravel().astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)    

        return frame
    
    def reset(self):
        self.state = STATES['INIT']
    
    def run(self):
        print("Running apriltag estimation")

        self.vid = cv2.VideoCapture(0)
        if self.record:
            width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
            height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
            size = (width, height)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.out = cv2.VideoWriter('output.mp4', fourcc, 20.0, size)
        if self.render:
            # does not work on mac
            # cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
            pass
        while True:
            ret, frame = self.vid.read()
            if not ret:
                print("Error: Could not read frame")
                break

            current_time = time.perf_counter()
            time_diff = current_time - self.vel["last_measurement"]
            # print("camera frequency", 1/(current_time - self.vel["last_measurement"]))
            self.vel["last_measurement"] = current_time
            
            tags = self.get_tags(frame)
            frame = self.render_tags(frame, tags)
            self.image = frame
            # if self.render:
            #     frame = self.render_tags(frame, tags)
            #     cv2.imshow('frame', frame)
            #     if cv2.waitKey(1) & 0xFF == ord('q'): 
            #         break
            
            if len(tags) > 0:
                if self.state == STATES['INIT']:
                    self.state = STATES['RUNNING']
                elif self.state == STATES['RUNNING']:
                    if self.ekf is not None:
                        # conversion from apriltags coordinate system to robot coordinate system (the coordinate system is different in apriltags)
                        # c = camera, r = robot
                        # tag = tags[0]

                        avg_global_coord_r = []
                        for tag in tags:
                            self.local_c_to_global_c = np.asarray(tag.pose_R)
                            self.global_c_to_local_c = self.local_c_to_global_c.T

                            self.global_c_to_local_r = self.camera_to_robot @ self.global_c_to_local_c
                            self.local_c_to_global_r = self.camera_to_robot @ self.local_c_to_global_c

                            self.local_r_to_global_r = self.camera_to_robot @ self.local_c_to_global_c @ self.camera_to_robot.T
                            self.global_r_to_local_r = self.camera_to_robot @ self.global_c_to_local_c @ self.camera_to_robot.T
                            
                            if cube_mode == "cube":
                                tag_to_robot_t = np.array([-0.16 / 2, 0, -0.16 / 2])
                            else:
                                tag_to_robot_t = np.array([-0.205 / 2, 0, -0.15])
                            global_coord_c = np.asarray(tag.pose_t).reshape(3)
                            global_coord_r = self.camera_to_robot @ global_coord_c + self.local_r_to_global_r @ tag_to_robot_t

                            avg_global_coord_r.append(global_coord_r)
                            
                        # rotation to correct tag orientation with respect to robot
                        if cube_mode == "cube":
                            if tag.tag_id == 3:
                                self.global_c_vel_to_local_r = self.camera_to_robot @ self.global_c_to_local_c
                            elif tag.tag_id == 0:
                                self.global_c_vel_to_local_r = rotation_z(np.pi/2) @ self.camera_to_robot @ self.global_c_to_local_c
                            elif tag.tag_id == 1:
                                self.global_c_vel_to_local_r = rotation_z(np.pi) @ self.camera_to_robot @ self.global_c_to_local_c
                            elif tag.tag_id == 2:
                                self.global_c_vel_to_local_r = rotation_z(-np.pi/2) @ self.camera_to_robot @ self.global_c_to_local_c
                        else:
                            if tag.tag_id == 0:
                                self.global_c_vel_to_local_r = self.camera_to_robot @ self.global_c_to_local_c
                            elif tag.tag_id == 1:
                                self.global_c_vel_to_local_r = rotation_z(np.pi/2) @ self.camera_to_robot @ self.global_c_to_local_c
                            elif tag.tag_id == 2:
                                self.global_c_vel_to_local_r = rotation_z(np.pi) @ self.camera_to_robot @ self.global_c_to_local_c
                            elif tag.tag_id == 3:
                                self.global_c_vel_to_local_r = rotation_z(-np.pi/2) @ self.camera_to_robot @ self.global_c_to_local_c
                                    
                            # self.global_c_vel_to_local_r = self.global_c_to_local_ro @ camera_to_robot.T
                            
                        # local_coord_r = self.global_c_to_local_r @ global_coord_c
                        global_coord_r = np.mean(avg_global_coord_r, axis=0)
                        if tracking_mode == "april_only":   
                            self.ekf.predict(None, time_diff)
                            self.ekf.update(global_coord_r)
                        elif tracking_mode == "acc_and_april_imu":
                            global_coord_imu = self.local_to_global_imu.T @ self.global_c_vel_to_local_r @ self.camera_to_robot.T @ global_coord_r
                            # for testing purposes
                            # self.ekf.predict(None, time_diff)
                            self.ekf.update(global_coord_imu)
                        elif tracking_mode == "acc_and_april_camera":
                            # for testing purposes
                            # self.ekf.predict(None, time_diff)
                            self.ekf.update(global_coord_r)
            
            if self.record:
                frame = self.render_tags(frame, tags)
                self.out.write(frame)
                    

        self.vid.release() 
        # Release the VideoWriter object
        self.out.release()
        # Destroy all the windows 
        cv2.destroyAllWindows()
    
    def set_imu_orientation(self, orientation):
        self.local_to_global_imu = orientation    
    
    def set_update_base_orientation(self):
        self.update_base_orientation = True
    
    def shutdown(self):
        self.vid.release() 
        # Release the VideoWriter object
        self.out.release()
        # Destroy all the windows 
        cv2.destroyAllWindows()
    