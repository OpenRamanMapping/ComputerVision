import cv2
import numpy as np

# Calibration parameters yaml file
camera_calibration_parameters_filename = 'calibration_chessboard2.yaml'

#ARUCO dictionary used for marker detection
ARUCO_DICT_CONFIG = cv2.aruco.DICT_4X4_50

ARUCO_MARKER_SIZE = 0.025

KALMAN_PROCESS_COEF = 5e-10              #Q
KALMAN_MEASUREMENT_COEF = 1e-8         #R
KALMAN_ERROR_COEF = 1

