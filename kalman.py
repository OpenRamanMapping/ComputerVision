import numpy as np
import cv2
from config import *
import scipy.signal
from collections import deque

# Define Kalman Filter
class PoseKalmanFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(6, 6)  # 6 state vars, 6 measurement vars

        # Transition matrix (assuming constant velocity model)
        self.kalman.transitionMatrix = np.eye(6, dtype=np.float32)

        # Measurement matrix
        self.kalman.measurementMatrix = np.eye(6, dtype=np.float32)

        # Process noise covariance
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * KALMAN_PROCESS_COEF

        # Measurement noise covariance
        self.kalman.measurementNoiseCov = np.eye(6, dtype=np.float32) * KALMAN_MEASUREMENT_COEF

        # Error covariance
        self.kalman.errorCovPost = np.eye(6, dtype=np.float32) * KALMAN_ERROR_COEF

        # Initial state (e.g., zeros)
        self.kalman.statePost = np.zeros(6, dtype=np.float32)

    def predict(self):
        return self.kalman.predict()

    def correct(self, measurement):
        return self.kalman.correct(measurement)

def smooth_pose_estimation(ids, rvecs, tvecs, pose_filter, pre_filter, post_filter):
    smoothed_poses = []
        
    for i in range(len(ids)):
        # Extract raw measurements (translation and rotation vectors)
        tvec = tvecs[i].flatten()
        rvec = rvecs[i].flatten()
        
        # Convert rotation vector to Euler angles (roll, pitch, yaw)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        euler_angles = cv2.decomposeProjectionMatrix(np.hstack((rotation_matrix, [[0], [0], [0]])))[-1]
        roll, yaw, pitch = euler_angles.flatten()
        
        median_yaw = yaw
        pre_filter.append(yaw)
        
        # Use median filter to get rid of outliers or spike
        medfilt_input = list(pre_filter)
        try:
            post_filter = (scipy.signal.medfilt(medfilt_input, kernel_size = 5))
        except Exception as e:
            print(e)
            
        # Save last angle after filtering
        if len(post_filter)>4:
            median_yaw = post_filter[4]
        
        # Construct measurement vector: [x, y, z, roll, pitch, yaw]
        measurement = np.array([tvec[0], tvec[1], tvec[2], roll, pitch, median_yaw], dtype=np.float32)

        # Predict the next state
        predicted = pose_filter.predict()

        # Update the Kalman filter with the current measurement
        corrected = pose_filter.correct(measurement)

        # Save the corrected pose for use
        smoothed_poses.append(corrected[:6])

    return smoothed_poses