import numpy as np
import cv2
import math as m

from config import *
from kalman import *

centre_of_frame = (int(FRAME_DIMENSIONS[0]/2),int(FRAME_DIMENSIONS[1]/2))

centered = False

pre_filter = deque(maxlen=5)
post_filter = deque(maxlen=5)

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        _, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(_)
    return rvecs, tvecs, trash

#ARUCO
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detectorParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)

cap = cv2.VideoCapture(4)

# Load the camera parameters from the saved file
cv_file = cv2.FileStorage(
    camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ) 
mtx = cv_file.getNode('K').mat()
dst = cv_file.getNode('D').mat()
cv_file.release()

contours = []

# initialize kalman filter
pose_filter = PoseKalmanFilter()

current_yaw = 0
previous_yaw = 0

yaw_mean = 0
yaw_mean_list = []
yaw_combined_list = []
yaw_corrected = []
yaw_real = []
translation_z = 15

marker_lost = False

# averaging distance
counter = 0
avg_translation = 0
avg_translation_counter = 0



def vect_mag(v:tuple):
    mag = m.sqrt(v[0]*v[0] + v[1]*v[1])
    return mag

while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame, FRAME_DIMENSIONS)
    
	# if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    lines = cv2.Canny(frame_gray, 80, 150)
   
    mask = cv2.adaptiveThreshold(
            frame_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3
        )
    
    try:
        contours, _ = cv2.findContours(lines, cv2.RETR_TREE ,cv2.CHAIN_APPROX_NONE)
        if contours:
            contour = max(contours, key = cv2.contourArea, default=0)
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), -1)

    except Exception as e:
        print(e)
        pass
   
    else:
        centered = False
    cv2.imshow('frame', frame)
    cv2.imshow('lines', lines)
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()