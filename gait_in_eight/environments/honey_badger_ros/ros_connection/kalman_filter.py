from filterpy.kalman import ExtendedKalmanFilter as EKF
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
import numpy as np

class AccelerationEKF(EKF):
    def __init__(self, dt, process_var, measurement_var, alpha=0.95):
        self.dim_x = 9  # state vector [x, y, z, vx, vy, vz, ax, ay, az]
        self.dim_z = 3  # measurement vector [ax, ay, az]
        self.process_var = process_var
        self.measurement_var = measurement_var
        
        super().__init__(dim_x=self.dim_x, dim_z=self.dim_z)
        
        self.dt = dt
        
        # Measurement function
        self.H = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        
        # Low-pass filter for acceleration
        self.lpf_x = LowPassFilter(alpha)
        self.lpf_y = LowPassFilter(alpha)
        self.lpf_z = LowPassFilter(alpha)
        
        self.reset()

    def reset(self):
        # Initial state
        self.x = np.zeros(self.dim_x)  # initial state [x, y, z, vx, vy, vz, ax, ay, az]

        # State transition matrix
        dt2 = 0.5 * self.dt ** 2
        self.F = np.array([[1, 0, 0, self.dt, 0, 0, dt2, 0, 0],
                           [0, 1, 0, 0, self.dt, 0, 0, dt2, 0],
                           [0, 0, 1, 0, 0, self.dt, 0, 0, dt2],
                           [0, 0, 0, 1, 0, 0, self.dt, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, self.dt, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, self.dt],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1]])

        # Process noise covariance
        q = self.process_var
        dt = self.dt
        q2 = 0.5 * dt ** 2 * q
        q3 = 0.5 * dt ** 3 * q
        q4 = 0.5 * dt ** 4 * q
        self.Q = np.array([[q4, 0, 0, q3, 0, 0, q2, 0, 0],
                           [0, q4, 0, 0, q3, 0, 0, q2, 0],
                           [0, 0, q4, 0, 0, q3, 0, 0, q2],
                           [q3, 0, 0, q2, 0, 0, dt * q, 0, 0],
                           [0, q3, 0, 0, q2, 0, 0, dt * q, 0],
                           [0, 0, q3, 0, 0, q2, 0, 0, dt * q],
                           [q2, 0, 0, dt * q, 0, 0, q, 0, 0],
                           [0, q2, 0, 0, dt * q, 0, 0, q, 0],
                           [0, 0, q2, 0, 0, dt * q, 0, 0, q]])

        # Measurement noise covariance
        self.R = np.eye(self.dim_z) * self.measurement_var

        # Initial covariance matrix
        self.P = np.eye(self.dim_x) * 0.1

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
    
    def update(self, z):
        # Apply low-pass filter to the measurement
        z_filtered = np.array([self.lpf_x.filter(z[0]), self.lpf_y.filter(z[1]), self.lpf_z.filter(z[2])])
        
        y = z_filtered - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.dim_x)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

# Low-pass filter class
class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.last_value = None

    def filter(self, value):
        if self.last_value is None:
            self.last_value = value
        else:
            self.last_value = self.alpha * value + (1 - self.alpha) * self.last_value
        return self.last_value


class LinearVelocityKF:
    def __init__(self, dt, process_var=10000, measurement_var=10000):
        self.dim_x = 6  # State vector: [x, vx, y, vy, z, vz]
        self.dim_z = 3  # Measurement vector: [x, y, z]
        
        self.dt = dt  # Time step
        self.process_var = process_var  # Process noise variance
        self.measurement_var = measurement_var  # Measurement noise variance
        self.measurement_var = 0.00134 ** 2
        self.process_var = 0.5e-3 ** 2


        self.reset_filter()

    def reset_filter(self):
        self.kf = KalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z)
        
        # Initial state (position and velocity in x, y, z)
        self.kf.x = np.zeros(self.dim_x)
        
        # State Transition matrix (F)
        self.kf.F = np.array([
            [1, self.dt, 0, 0, 0, 0],  # x, vx
            [0, 1, 0, 0, 0, 0],        # vx
            [0, 0, 1, self.dt, 0, 0],  # y, vy
            [0, 0, 0, 1, 0, 0],        # vy
            [0, 0, 0, 0, 1, self.dt],  # z, vz
            [0, 0, 0, 0, 0, 1]         # vz
        ])
        
        # Control Input matrix (B) for acceleration input
        self.kf.B = np.array([
            [0.5 * self.dt ** 2, 0, 0],   # x
            [self.dt, 0, 0],              # vx
            [0, 0.5 * self.dt ** 2, 0],   # y
            [0, self.dt, 0],              # vy
            [0, 0, 0.5 * self.dt ** 2],   # z
            [0, 0, self.dt]               # vz
        ])
        
        # Measurement matrix (H) for position measurement
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],  # x
            [0, 0, 1, 0, 0, 0],  # y
            [0, 0, 0, 0, 1, 0]   # z
        ])
        
        # Process noise covariance matrix (Q)
        q = self.process_var
        self.kf.Q = np.array([
            [q, 0, 0, 0, 0, 0],
            [0, q, 0, 0, 0, 0],
            [0, 0, q, 0, 0, 0],
            [0, 0, 0, q, 0, 0],
            [0, 0, 0, 0, q, 0],
            [0, 0, 0, 0, 0, q]
        ])
        
        # Measurement noise covariance matrix (R)
        self.kf.R = np.array([
            [self.measurement_var, 0, 0],
            [0, self.measurement_var, 0],
            [0, 0, self.measurement_var]
        ])
        
        # Initial state covariance matrix (P)
        self.kf.P = np.eye(self.dim_x)
    
    def update_matrices(self, dt):
        self.kf.F = np.array([
            [1, dt, 0, 0, 0, 0],  # x, vx
            [0, 1, 0, 0, 0, 0],   # vx
            [0, 0, 1, dt, 0, 0],  # y, vy
            [0, 0, 0, 1, 0, 0],   # vy
            [0, 0, 0, 0, 1, dt],  # z, vz
            [0, 0, 0, 0, 0, 1]    # vz
        ])
        
        self.kf.B = np.array([
            [0.5 * dt ** 2, 0, 0],  # x
            [dt, 0, 0],             # vx
            [0, 0.5 * dt ** 2, 0],  # y
            [0, dt, 0],             # vy
            [0, 0, 0.5 * dt ** 2],  # z
            [0, 0, dt]              # vz
        ])

    def predict(self, acceleration=None, dt = None):
        """Predicts the next state based on the current state and control input (acceleration)."""
        if dt is not None:
            self.update_matrices(dt)
        if acceleration is not None:
            self.kf.predict(u=acceleration)
        else:
            self.kf.predict()
    
    def update(self, position):
        """Updates the state with a new position measurement."""
        self.kf.update(position)
    
    def get_state(self):
        """Returns the current estimated state [x, vx, y, vy, z, vz]."""
        return self.kf.x
