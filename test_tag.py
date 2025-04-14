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
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
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

    # Our operations on the frame come here
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.rectangle(frame, (centre_of_frame[0] - MARGIN_OF_CENTER , centre_of_frame[1] - MARGIN_OF_CENTER), (centre_of_frame[0] + MARGIN_OF_CENTER , centre_of_frame[1] + MARGIN_OF_CENTER), color_of_center, 3)

    corners, ids, rejectedImgPoints = detector.detectMarkers(frame_gray)
    frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(255, 0, 0))

    if ids is not None:
        ids = ids.flatten()
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(255, 0, 0))

        #c1 v1-> c2
        ##########
        #        #
        #        #
        #        #
        ##########
        #c4 <-v3 c3

        corner_1 = (int(corners[0][0][0][0]), int(corners[0][0][0][1]))
        corner_2 = (int(corners[0][0][1][0]), int(corners[0][0][1][1]))
        corner_3 = (int(corners[0][0][2][0]), int(corners[0][0][2][1]))
        corner_4 = (int(corners[0][0][3][0]), int(corners[0][0][3][1]))

        centre_x1 = (corner_1[0]+corner_2[0])/2
        centre_x2 = (corner_3[0]+corner_4[0])/2
        centre_y1 = (corner_1[1]+corner_2[1])/2
        centre_y2 = (corner_3[1]+corner_4[1])/2

        #Center of ArUco marker
        centre = (int((centre_x1+centre_x2)/2), int((centre_y1+centre_y2)/2))
        
        vect12 = (corner_2[0] - corner_1[0], corner_2[1] - corner_1[1])
        vect23 = (corner_2[0] - corner_3[0], corner_2[1] - corner_3[1])
        vect34 = (corner_4[0] - corner_3[0], corner_4[1] - corner_3[1])
        vect41 = (corner_4[0] - corner_1[0], corner_4[1] - corner_1[1])
        
        vectors = [vect12, vect23, vect34, vect41]

        mags = [vect_mag(v) for v in vectors]

        ratios = [mag / ARUCO_MARKER_SIZE for mag in mags]

        theta_23 = m.asin(-vect23[1]/mags[1])
        theta_12 = m.asin(vect12[1]/mags[0])
        theta_41 = m.asin(-vect41[1]/mags[3])

        p1 = (int(corner_3[0] + 0.13*ratios[1]*m.cos(theta_23)), int(corner_3[1] - 0.13*m.sin(theta_23)*ratios[1]))
        p2 = (int(p1[0] - 0.1*ratios[0]*m.cos(theta_12)), int(p1[1] - 0.1*ratios[0]*m.sin(theta_12)))
        p3 = (int(p2[0] - 0.1*ratios[3]*m.cos(theta_41)), int(p2[1] - 0.1*ratios[3]*m.sin(theta_41)))

        #cv2.line(frame, corner_2, p1, (0, 255, 0), 5, cv2.LINE_AA)
        #cv2.line(frame, p1, p2, (0, 255, 0), 5, cv2.LINE_AA)
        #cv2.line(frame, p2, p3, (0, 255, 0), 5, cv2.LINE_AA)
        #cv2.line(frame, p3, corner_2, (0, 255, 0), 5, cv2.LINE_AA)

        vertices = [
            corner_2,
            p1,
            p2,
            p3
        ]

        vertices = np.array([vertices], np.int32)
        lines = cv2.Canny(frame, 50, 150)
        
        mask_black = np.zeros_like(lines)

        cv2.fillPoly(mask_black, vertices, (255, 255, 255))

        frame_lines = cv2.bitwise_and(mask_black, lines)
        
        try:
            contours, hierarchy = cv2.findContours(frame_lines, cv2.RETR_TREE ,cv2.CHAIN_APPROX_NONE)
        except Exception as e:
            print(e)
            
        #if contours:
          #  contour = max(contours, key = cv2.contourArea, default=0)
         #   cv2.drawContours(frame, [contour], -1, (0, 255, 0), -1)

        if centre[0] > centre_of_frame[0] - MARGIN_OF_CENTER and centre[0] < centre_of_frame[0] + MARGIN_OF_CENTER and centre[1] > centre_of_frame[1] - MARGIN_OF_CENTER and centre[1] < centre_of_frame[1] + MARGIN_OF_CENTER:
            centered = True
        else:
            centered = False

            if ids is not None:
                marker_lost = False
            #  Get the rotation and translation vectors
        rvecs, tvecs, obj_points = my_estimatePoseSingleMarkers(
        corners, ARUCO_MARKER_SIZE, mtx, dst
        )

        # smoothen using kalman filter and median for yaw
        smoothed_poses = smooth_pose_estimation(
            ids, rvecs, tvecs, pose_filter, pre_filter, post_filter
        )

        for i, marker_id in enumerate(ids):
                smoothed_pose = smoothed_poses[i]
                tvec = smoothed_pose[:3]
                roll, pitch, yaw = smoothed_pose[3:]

                # round values to 2 decimal places
                roll = round(roll, 2)
                pitch = round(pitch, 2)
                yaw = round(yaw, 2)

                # distances from tag in cm
                transform_translation_x = tvec[0] * 1000
                transform_translation_y = tvec[1] * 1000
                translation_z = tvec[2] * 1000 - 16

                # ~ cv2.putText(frame, f"translation z: {translation_z:.2f}", (0, 150),
                             # ~ cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                # ~ cv2.putText(frame, f"yaw deg: {yaw_deg:.2f}", (0, 200),
                             # ~ cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                
                cv2.drawFrameAxes(frame, mtx, dst, rvecs[i], tvecs[i], 0.05) 
                
                avg_translation_counter += translation_z
                counter +=1
                if counter % 20 == 0:
                    avg_translation = avg_translation_counter / 20
                    #print(f'Z translation: {avg_translation:.2f}, Roll: {roll:.2f}, Pitch: {pitch:.2f}, Yaw: {yaw:.2f}')
                    avg_translation_counter, avg_translation = [0, 0]
        else:
            marker_lost = True  
        
        cv2.putText(frame, ".", centre, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
    else:
        centered = False
    cv2.imshow('frame', frame)
    #cv2.imshow('lines', frame_lines)
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()