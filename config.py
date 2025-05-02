import cv2
import numpy as np

# Calibration parameters yaml file
CAMERA_CALIBRATION_PARAMETERS_FILENAME = 'calibration_chessboard2.yaml'

#user-selected distance between sample points
GAP_MM = 5

#ARUCO dictionary used for marker detection
ARUCO_DICT_CONFIG = cv2.aruco.DICT_4X4_50

ARUCO_MARKER_SIZE = 0.025

