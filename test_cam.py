import numpy as np
import cv2
import math as m
from imutils import perspective
from config import *
from pprint import pprint

#initiate ARUCO detection objects
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detectorParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)

home_position = (266, 71)

#initiate camera stream
cap = cv2.VideoCapture(4)

# Load the camera calibration parameters from the saved file
cv_file = cv2.FileStorage(
    CAMERA_CALIBRATION_PARAMETERS_FILENAME, cv2.FILE_STORAGE_READ) 
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

def get_lines(image, theshold_low = 200, threshold_high = 250):
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
        self.mid_marker0 = [int(p) for p in midpoint(corners[0][0][0], corners[0][0][1])]
        self.mid_marker1 = [int(p) for p in midpoint(corners[0][0][1], corners[0][0][2])]
        self.mid_marker2 = [int(p) for p in midpoint(corners[0][0][2], corners[0][0][3])]
        self.mid_marker3 = [int(p) for p in midpoint(corners[0][0][3], corners[0][0][0])]
        cv2.circle(frame, self.mid_marker1, 2, (255, 0, 0), -1)
        cv2.circle(frame, self.mid_marker2, 2, (255, 0, 0), -1)
        cv2.circle(frame, self.mid_marker3, 2, (255, 0, 0), -1)
        cv2.circle(frame, self.mid_marker0, 2, (255, 0, 0), -1)
        
        perimeter = self.size * 4
        
        pix_perim = cv2.arcLength(corners[0], True)
        
        ratio = perimeter/pix_perim
        
        return ratio
    def get_center(self):
        '''
        Returns center position of ArUco marker
        '''
        center = (int((self.mid_marker0[0]+self.mid_marker2[0])/2), int((self.mid_marker1[1]+self.mid_marker3[1])/2))

        return center
        
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
        ret, frame_orig = cap.read()
        
        #cv2.imwrite('frame_old1.jpg', frame_orig)

        h = frame_orig.shape[0]
        w = frame_orig.shape[1]
        cv2.circle(frame_orig, (int(w/2), int(h/2)), 3, (255, 0, 255), -1)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dst, (w, h), 1, (w, h))

        frame = cv2.undistort(frame_orig, mtx, dst, None, newcameramtx)
        #frame = frame_orig

        #Operation on the frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_blur = cv2.bilateralFilter(frame_gray, 9, 40, 40)

        #get corners of ArUco
        corners, ids, rejectedImgPoints = detector.detectMarkers(frame_gray)

        if ids is not None:
            ids = ids.flatten()
            marker = Marker(corners, ARUCO_MARKER_SIZE)
            ratio = marker.get_ratio()
            center = marker.get_center()
            print(center)
            difference_mm = ((home_position[0]-center[0])*1000*ratio, (home_position[1]-center[1])*1000*ratio)
            print(difference_mm)
            cv2.circle(frame, center, 2, (0, 255, 255), -1)
            #print(corners[0][0])
            
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()