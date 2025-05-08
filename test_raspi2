import numpy as np
import cv2
import math as m
from imutils import perspective
from config import *

from picamera2 import Picamera2
import time

from libcamera import controls, Transform

import os

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main = {"size": (640, 480)},transform = Transform(hflip=1, vflip=1)))
picam2.start()

#initiate ARUCO detection objects
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detectorParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)

# Load the camera calibration parameters from the saved file
cv_file = cv2.FileStorage(
    "calibration_raspi.yaml", cv2.FILE_STORAGE_READ) 
mtx = cv_file.getNode('K').mat()
dst = cv_file.getNode('D').mat()
cv_file.release()

contours = []

def distance(pt1, pt2):
    '''
    Returns euclidean distance between two points.
    '''
    v = (pt2[0] - pt1[0], pt2[1] - pt1[1])
    return m.sqrt(v[0]*v[0] + v[1]*v[1])

def midpoint(ptA, ptB):
    '''
    Returns midpoint between two points.
    '''
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def prep_frame_lines(plate, frame_blur, frame_gray):
    '''
    Returns a frame prepped for finding lines of sample.
    '''
    plate_contours, _ = cv2.findContours(plate, cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)
    if plate_contours:
        plate_contour = max(plate_contours, key = cv2.contourArea, default=0) 
        approx = cv2.approxPolyDP(plate_contour, 10, closed=True)

        plate_vertices = cv2.boundingRect(plate_contour)
        
        mask_black = np.zeros_like(frame_blur)
        vertices = np.array([approx], np.int32)
        cv2.fillPoly(mask_black, vertices, 255)
        frame_lines = cv2.bitwise_and(frame_gray, mask_black)
    return frame_lines, plate_vertices

def get_lines(image, theshold_low = 100, threshold_high = 120):
    '''
    Returns lines from a grayscale frame.
    '''
    lines = cv2.Canny(image, theshold_low, threshold_high)
    lines = cv2.dilate(lines, None)
    lines = cv2.erode(lines, None)
    return lines

class Marker:
    def __init__(self, corners, size):
        self.corners = corners
        self.size = size

    def get_ratio(self):
        '''
        Returns ratio of pixels to mm
        '''
        mid_marker0 = [int(p) for p in midpoint(corners[0][0][0], corners[0][0][1])]
        mid_marker1 = [int(p) for p in midpoint(corners[0][0][1], corners[0][0][2])]
        mid_marker2 = [int(p) for p in midpoint(corners[0][0][2], corners[0][0][3])]
        mid_marker3 = [int(p) for p in midpoint(corners[0][0][3], corners[0][0][0])]
        cv2.circle(frame, mid_marker1, 2, (255, 0, 0), -1)
        cv2.circle(frame, mid_marker2, 2, (255, 0, 0), -1)
        cv2.circle(frame, mid_marker3, 2, (255, 0, 0), -1)
        cv2.circle(frame, mid_marker0, 2, (255, 0, 0), -1)
        
        perimeter = self.size * 4
        
        pix_perim = cv2.arcLength(corners[0], True)
        
        ratio = perimeter/pix_perim
        
        return ratio

class Sample:
    def __init__(self, contour):
        self.contour = contour

    def get_size(self):
        '''
        Returns size of sample in mm.
        '''
        rect = cv2.minAreaRect(self.contour)

        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (p1, p2, p3, p4) = box

        mid_0 = [int(p) for p in midpoint(p1, p2)]
        mid_1 = [int(p) for p in midpoint(p2, p3)]
        mid_2 = [int(p) for p in midpoint(p3, p4)]
        mid_3 = [int(p) for p in midpoint(p1, p4)]

        dist_0 = distance(mid_0, mid_2)
        dist_1 = distance(mid_1, mid_3)

        size1 = dist_0*ratio*1000
        size2 = dist_1*ratio*1000
        size = (size1, size2)

        self.size = size
        self.box = box
        self.midpoints = [mid_0, mid_1, mid_2, mid_3]
        
        return size
    
    def draw_cont(self, real_size, frame):
        '''
        Draws contours and dimensions of sample on frame
        '''
        size = self.size
        size = sorted(size)
        error = (abs(real_size[0]-size[0])/real_size[0] + abs(real_size[1]-size[1])/real_size[1])/2*100
        
        print(f'size: {size[0]:10.2f} mm, {size[1]:10.2f} mm, error: {error:10.1f}')

        cv2.rectangle(frame, (self.midpoints[0][0], self.midpoints[0][1]-8), (self.midpoints[0][0]+300, self.midpoints[0][1]-30), (0, 0, 0), -1)
        cv2.putText(frame, f'Size: {size[0]:.1f}, {size[1]:.1f} Error: {error:.1f}', (self.midpoints[0][0], self.midpoints[0][1]-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.circle(frame, self.midpoints[0], 3, (0, 255, 0), -1)
        cv2.circle(frame, self.midpoints[1], 3, (0, 255, 0), -1)
        cv2.circle(frame, self.midpoints[2], 3, (0, 255, 0), -1)
        cv2.circle(frame, self.midpoints[3], 3, (0, 255, 0), -1)

        cv2.drawContours(frame, [self.box.astype("int")], -1, (255, 255, 0),1)

if __name__ == "__main__":

    while True:
        frame_orig = picam2.capture_array()
        cv2.imwrite("sample_pic_rpi.jpg", frame_orig)
        #frame_orig = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h = frame_orig.shape[0]
        w = frame_orig.shape[1]

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dst, (w, h), 1, (w, h))

        #frame = cv2.undistort(frame_orig, mtx, dst, None, newcameramtx)
        frame = frame_orig

        #Operation on the frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_blur = cv2.bilateralFilter(frame_gray, 9, 40, 40)

        #get corners of ArUco
        corners, ids, rejectedImgPoints = detector.detectMarkers(frame_gray)

        if ids is not None:
            ids = ids.flatten()
            marker = Marker(corners, ARUCO_MARKER_SIZE)
            ratio = marker.get_ratio()
            #print(corners[0][0])

        plate = cv2.adaptiveThreshold(
            frame_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 111, 15)
        
        try: 
            frame_lines, (x_box, y_box, w_box, h_box) = prep_frame_lines(plate, frame_blur, frame_gray)
            
            lines = get_lines(frame_lines)
            
        except Exception as e:
            print(e)

        if lines.any():
            cv2.imshow('lines2', lines)
            try:
                contours, hierarchy = cv2.findContours(lines, cv2.RETR_TREE ,cv2.CHAIN_APPROX_NONE)
                if contours:
                    #Filter out contours too large/too small
                    for c in contours:
                        print(cv2.contourArea(c))
                    contours_filtered = [c for c in contours if 500<cv2.contourArea(c)<20000]
                    #Filter contours by size
                    contours_sorted = sorted(contours_filtered, key = cv2.contourArea, reverse=True)
                    #Grab the first (largest) contour
                    c = contours_sorted[0]

                    #Create sample object based on contour
                    sample = Sample(c)

                    #Measure size of object in mm
                    size = sample.get_size()
                    #Draw bounding rectangle of object and print size and error
                    sample.draw_cont(real_size=(41.7, 47.14), frame = frame)
                
                    mask_black_cont = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(mask_black_cont, [c], -1, color = 1, thickness=-1)

                    #frame_show = frame[y_box:y_box+h_box, x_box:x_box+w_box]

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
                    #pprint(contour_ones)
                    for i in contour_ones:
                        frame[i] = (0, 255, 0)
                    
            except Exception as e:
                print('cont',e)
                pass
            
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cv2.destroyAllWindows()
