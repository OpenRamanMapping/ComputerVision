import numpy as np
import cv2
import math as m

from config import *
from kalman import *

centered = False

pre_filter = deque(maxlen=5)
post_filter = deque(maxlen=5)

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
median_roll = []
# averaging distance
counter = 0
avg_translation = 0
avg_translation_counter = 0
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

def vect_mag(v:tuple):
    mag = m.sqrt(v[0]*v[0] + v[1]*v[1])
    return mag

def get_corners(corners):
    '''
    Create a list of corners of the marker in tuples of int values.
    '''
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

    id_corners = [corner_1, corner_2, corner_3, corner_4]
    
    return id_corners

def get_vectors(id_corners):
    '''
    Calculate vectors from corners of a marker.
    '''
    vect12 = (id_corners[1][0] - id_corners[0][0], id_corners[1][1] - id_corners[0][1])
    vect23 = (id_corners[1][0] - id_corners[2][0], id_corners[1][1] - id_corners[2][1])
    vect34 = (id_corners[3][0] - id_corners[2][0], id_corners[3][1] - id_corners[2][1])
    vect41 = (id_corners[3][0] - id_corners[0][0], id_corners[3][1] - id_corners[0][1])
    
    vectors = [vect12, vect23, vect34, vect41]

    return vectors

while True:

    ret, frame = cap.read()
    
	# if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.bilateralFilter(frame_gray, 9, 40, 40)

    corners, ids, rejectedImgPoints = detector.detectMarkers(frame_gray)

    if ids is not None:
        
        ids = ids.flatten()
        
        frame_ids = cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(255, 0, 0))

        id_corners = get_corners(corners)

        vectors = get_vectors(id_corners)

        mags = [vect_mag(v) for v in vectors]

        ratios = [mag / ARUCO_MARKER_SIZE for mag in mags]

        #  Get the rotation and translation vectors
        rvecs, tvecs, obj_points = my_estimatePoseSingleMarkers(
        corners, ARUCO_MARKER_SIZE, mtx, dst
        )

        # smoothen using kalman filter and median
        smoothed_poses = smooth_pose_estimation(
            ids, rvecs, tvecs, pose_filter, pre_filter, post_filter
        )

        for i, marker_id in enumerate(ids):
            smoothed_pose = smoothed_poses[i]
            tvec = tvecs[i]
            rvec = rvecs[i]
            roll, pitch, yaw = rvecs[i]
            # round values to 2 decimal places
            # roll = round(roll[0], 2)
            # pitch = round(pitch[0], 2)
            # yaw = round(yaw[0], 2)

            print('1:', roll, pitch, yaw)
            # distances from tag in cm
            transform_translation_x = tvec[0] * 1000
            transform_translation_y = tvec[1] * 1000
            translation_z = tvec[2][0] * 1000

            length = 0.1  # in meters (adjust to match your marker units)
            marker_offset = 0.01

            pre_filter.append(roll[0])
            
            # Use median filter to get rid of outliers or spike
            medfilt_input = list(pre_filter)
            
            try:
                if len(medfilt_input)>4:
                    post_filter = (scipy.signal.medfilt(medfilt_input, kernel_size = 5))
            except Exception as e:
                print(e)
                
            # Save last angle after filtering
            if len(post_filter)>4:
                median_roll = post_filter[4] if post_filter[4]!=0 else post_filter[3]
                rvec[0][0] = median_roll
            
            # Starting point in 3D 
            start_1 = np.array([[-marker_offset, -marker_offset, 0]], dtype=np.float32)
            end_1 = np.array([[-marker_offset, -marker_offset-0.13, 0]], dtype=np.float32)
            end_2 = np.array([[length, -marker_offset-0.13, 0]], dtype=np.float32)
            end_3 = np.array([[length-marker_offset, -marker_offset, 0]], dtype=np.float32)

            # Project both points to 2D image
            pts_1, _ = cv2.projectPoints(np.vstack([start_1, end_1]), rvec, tvec, mtx, dst)
            pts_2, _ = cv2.projectPoints(np.vstack([start_1, end_2]), rvec, tvec, mtx, dst)
            pts_3, _ = cv2.projectPoints(np.vstack([start_1, end_3]), rvec, tvec, mtx, dst)
            
            pt1 = tuple(pts_1[0].ravel().astype(int))
            pt2 = tuple(pts_1[1].ravel().astype(int))
            pt3 = tuple(pts_2[1].ravel().astype(int))
            pt4 = tuple(pts_3[1].ravel().astype(int))
            
            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
            cv2.line(frame, pt2, pt3, (0, 255, 255), 2)
            cv2.line(frame, pt3, pt4, (0, 255, 255), 2)
            cv2.line(frame, pt4, pt1, (0, 255, 255), 2)

            vertices = [
                pt1, 
                pt2, 
                pt3,
                pt4
            ]
            
            mask_black = np.zeros_like(frame_gray)

            vertices = np.array([vertices], np.int32)
            cv2.fillPoly(mask_black, vertices, 255)
            frame_lines = cv2.bitwise_and(mask_black, frame_blur)
            
            lines = cv2.Canny(frame_lines, 200, 300)
            lines = cv2.dilate(lines, np.ones((5, 5), np.uint8))
            #lines = cv2.erode(lines, np.ones((3, 3), np.uint8))

            if lines.any():
                cv2.drawFrameAxes(frame, mtx, dst, rvecs[i], tvecs[i], 0.05) 
                avg_translation_counter += translation_z

                counter +=1
                if counter % 20 == 0:
                    avg_translation = avg_translation_counter / 20
                    print(f'Z translation: {avg_translation:.2f}')
                    avg_translation_counter, avg_translation = [0, 0]

                try:
                    contours, hierarchy = cv2.findContours(lines, cv2.RETR_TREE ,cv2.CHAIN_APPROX_NONE)
                    if contours:
                        contour = max(contours, key = cv2.contourArea, default=0)
                       # cv2.drawContours(frame, [contour], -1, (0, 255, 0), -1)
                        M = cv2.moments(contour)
                        
                        #for c in contour[0]:
                        #   cont_coord = (int(c[0]/640*100), int(c[0]/360*100))
                                    
                        #cv2.putText(frame, f'  .edge {contour[0][0]}', contour[0][0], cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                        #print(f'contour: {contour} scaled: {cont_coord}')         

                except Exception as e:
                    print(e)
                    pass
                cv2.imshow('lines', frame_lines)    

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()