import numpy as np
import cv2
import math as m
from imutils import perspective
from config import *
from kalman import *
from pprint import pprint

from timeit import default_timer as timer

#ARUCO
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detectorParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)

#cap = cv2.VideoCapture(4)

# Load the camera parameters from the saved file
cv_file = cv2.FileStorage(
    camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ) 
mtx = cv_file.getNode('K').mat()
dst = cv_file.getNode('D').mat()
cv_file.release()

contours = []

def distance(pt1, pt2):
    v = (pt2[0] - pt1[0], pt2[1] - pt1[1])
    return m.sqrt(v[0]*v[0] + v[1]*v[1])

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

GAP_MM = 1 #gap between measurement points in mm

frame_old = cv2.imread(r'frame_old1.jpg')
#cv2.imshow('frame_orig', frame_old)
h = frame_old.shape[0]
w = frame_old.shape[1]

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
    mtx, dst, (w, h), 1, (w, h))

frame = cv2.undistort(frame_old, mtx, dst, None, newcameramtx)

#frame = frame_old

# operations on the frame come here
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame_blur = cv2.bilateralFilter(frame_gray, 9, 40, 40)

corners, ids, rejectedImgPoints = detector.detectMarkers(frame_gray)

#cv2.aruco.drawDetectedMarkers(frame, corners, ids)

if ids is not None:
    ids = ids.flatten()
    
    mid_marker0 = [int(p) for p in midpoint(corners[0][0][0], corners[0][0][1])]
    mid_marker1 = [int(p) for p in midpoint(corners[0][0][1], corners[0][0][2])]
    mid_marker2 = [int(p) for p in midpoint(corners[0][0][2], corners[0][0][3])]
    mid_marker3 = [int(p) for p in midpoint(corners[0][0][3], corners[0][0][0])]
    
    cv2.circle(frame, mid_marker0, 2, (255, 0, 0), -1)
    cv2.circle(frame, mid_marker1, 2, (255, 0, 0), -1)
    cv2.circle(frame, mid_marker2, 2, (255, 0, 0), -1)
    cv2.circle(frame, mid_marker3, 2, (255, 0, 0), -1)
    
    perimeter = ARUCO_MARKER_SIZE * 4
    print('p',distance(mid_marker0, mid_marker2))
    pix_perim = cv2.arcLength(corners[0], True)
    #ratio mm/pixels
    ratio = perimeter/pix_perim 

contour_pts = []

plate = cv2.adaptiveThreshold(
    frame_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 111, 15)

try:
    plate_contours, hierarchy = cv2.findContours(plate, cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)
    if plate_contours:
        plate_contour = max(plate_contours, key = cv2.contourArea, default=0) 
        approx = cv2.approxPolyDP(plate_contour, 10, closed=True)
        
        mask_black = np.zeros_like(frame_blur)
        vertices = np.array([approx], np.int32)
        cv2.fillPoly(mask_black, vertices, 255)
        frame_lines = cv2.bitwise_and(frame_gray, mask_black)

except Exception as e:
    print(e)

lines = cv2.Canny(frame_gray, 200, 300)
lines = cv2.dilate(lines, None)
lines = cv2.erode(lines, None)

if lines.any():

    try:
        contours, hierarchy = cv2.findContours(lines, cv2.RETR_TREE ,cv2.CHAIN_APPROX_NONE)
        #print("hierarchy: ", hierarchy)

        if contours:
            #Filter out contours too large/too small
            contours_filtered = [c for c in contours if 500<cv2.contourArea(c)<15000]
            #Filter contours by size
            contours_sorted = sorted(contours_filtered, key = cv2.contourArea, reverse=True)
            #Grab the first (largest) contour
            c = contours_sorted[0]

            rect = cv2.minAreaRect(c)

            box = cv2.boxPoints(rect)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            time_start = timer()

            # x_p, y_p, w_p, h_p = cv2.boundingRect(c)
            # for i in range(x_p, x_p + w_p):
            #     for j in range(y_p, y_p + h_p):
            #         pt = (i, j)
            #         test_pts = cv2.pointPolygonTest(c, pt, measureDist=False)
            #         if test_pts >-1:
            #             contour_pts.append(pt)
            # print(timer() - time_start)

            # for i in range(w):
            #     for j in range(h):
            #         pt = (i, j)
            #         test_pts = cv2.pointPolygonTest(c, pt, measureDist=False)
            #         if test_pts >-1:
            #             contour_pts.append(pt)
            # print(timer() - time_start)

            mask_black_cont = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask_black_cont, [c], -1, color = 1, thickness=-1)

            #print(timer() - time_start)
            
            if GAP_MM <= 0:
                raise ValueError("gap cannot be 0 or less, choose continuous scanning instead")

            gap_pix = int(GAP_MM/1000/ratio)
           
            mask_scan = np.zeros((h, w), dtype=np.uint8)

            for i in range(0, h, gap_pix):
                for j in range(0, w, gap_pix):
                    index = (i, j)
                    if mask_black_cont[index]:
                        mask_scan[index] = 1                    

            y_shape, x_shape = np.where(mask_scan == 1)
            contour_ones = list(zip(y_shape, x_shape))     

            contour_ones = ([(int(c[0]), int(c[1])) for c in contour_ones])

            print(contour_ones)
            #cv2.drawContours(frame, [contour_pts], -1, color = (0, 255, 0), thickness=-1)   

            for i in contour_ones:
                frame[i] = (0, 255, 0)

            # for index in contour_pts:
            #     #frame[index] = (0, 0, 0)
            #     mask_black_cont[index] = 0
            #     counter_pix += 1
            #     if counter_pix % gap_pix == 0:
            #         mask_black_cont[index] = 1
            #         frame[index] = (255, 0, 0)       

            # cv2.imshow("mask", mask_black_cont)

            (p1, p2, p3, p4) = box

            mid_0 = [int(p) for p in midpoint(p1, p2)]
            mid_1 = [int(p) for p in midpoint(p2, p3)]
            mid_2 = [int(p) for p in midpoint(p3, p4)]
            mid_3 = [int(p) for p in midpoint(p1, p4)]

            obj_pts = np.array([[mid_0], [mid_1], [mid_2], [mid_3]], dtype=np.float32)
            
            #dist_0 = distance(mid_00[0], mid_02[0])
            #dist_1 = distance(mid_01[0], mid_03[0])

            dist_0 = distance(mid_0, mid_2)
            dist_1 = distance(mid_1, mid_3)

            size1 = dist_0*ratio*1000
            size2 = dist_1*ratio*1000

            real_size = (41.6, 48)

            error = (abs(real_size[0]-size1)/real_size[0] + abs(real_size[1]-size2)/real_size[1])/2*100
            
            print(f'size: {size1:10.2f} mm, {size2:10.2f} mm, error: {error:10.1f}')

            cv2.rectangle(frame, (mid_0[0], mid_0[1]+5), (mid_0[0]+200, mid_0[1]-20), (0, 0, 0), -1)
            cv2.putText(frame, f'Size: {size1:.1f}, {size2:.1f}', mid_0, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            cv2.circle(frame, mid_0, 3, (0, 255, 0), -1)
            cv2.circle(frame, mid_1, 3, (0, 255, 0), -1)
            cv2.circle(frame, mid_2, 3, (0, 255, 0), -1)
            cv2.circle(frame, mid_3, 3, (0, 255, 0), -1)

            cv2.drawContours(frame, [box.astype("int")], -1, (255, 255, 0),1)

    except Exception as e:
        print('cont',e)
        pass

cv2.imshow('frame', frame)

cv2.waitKey(0)
cv2.destroyAllWindows()