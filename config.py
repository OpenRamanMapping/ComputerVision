import cv2
import numpy as np

FRAME_DIMENSIONS = (640, 360)
WIDTH_OF_IMAGE = FRAME_DIMENSIONS[0]
HEIGHT_OF_IMAGE = FRAME_DIMENSIONS[1]
CENTRE_OF_FRAME = (int(FRAME_DIMENSIONS[0]/2),int(FRAME_DIMENSIONS[1]/2))

# Calibration parameters yaml file
camera_calibration_parameters_filename = 'calibration_chessboard.yaml'

#ARUCO dictionary used for marker detection
ARUCO_DICT_CONFIG = cv2.aruco.DICT_4X4_50

# percentage of frame for black line to be sought
LINE_SEEKING_PORTION = 20

ARUCO_MARKER_SIZE = 0.025

# margins of error for marker alignment
MARGIN_OF_CENTER = 40
MARGIN_OF_DISTANCE = 2
MARGIN_OF_ANGLE = 2

DISTANCE_FROM_MARKER = 13

SPEED_STRAFE = 15
SPEED_DRIFT = 15
SPEED_DRIVE = 11
SPEED_SPIN = 11

MAX_SPEED = 20
MIN_SPEED = 15

SPEED_COEF = 0.5
OFFSET_COEF = 0.35
ANGLE_COEF = 0.15

KALMAN_PROCESS_COEF = 5e-10              #Q
KALMAN_MEASUREMENT_COEF = 1e-8         #R
KALMAN_ERROR_COEF = 1

