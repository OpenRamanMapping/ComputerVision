import numpy as np
import cv2
import math as m
from imutils import perspective
from config import *

from picamera2 import Picamera2
import time

from libcamera import controls, Transform

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main = {"size": (640, 480)},transform = Transform(hflip=1, vflip=1)))
picam2.start()
import cv2
import sys
import numpy as np
import cv2
import math as m
from imutils import perspective
from config import *
from PyQt5 import  QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QMainWindow
from PyQt5.QtSerialPort import QSerialPort, QSerialPortInfo
from PyQt5.QtCore import QIODevice, QByteArray, pyqtSlot, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from qtwidgets import Toggle, AnimatedToggle
import time 

#initiate ARUCO detection objects
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detectorParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)

# Load the camera calibration parameters from the saved file
cv_file = cv2.FileStorage(
    CAMERA_CALIBRATION_PARAMETERS_FILENAME, cv2.FILE_STORAGE_READ) 
mtx = cv_file.getNode('K').mat()
dst = cv_file.getNode('D').mat()
cv_file.release()

contours = []
lines = []

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
    def __init__(self, corners, size, frame):
        self.corners = corners
        self.size = size
        self.frame = frame

    def get_ratio(self):
        corners = self.corners
        '''
        Returns ratio of pixels to mm
        '''
        self.mid_marker0 = [int(p) for p in midpoint(corners[0][0][0], corners[0][0][1])]
        self.mid_marker1 = [int(p) for p in midpoint(corners[0][0][1], corners[0][0][2])]
        self.mid_marker2 = [int(p) for p in midpoint(corners[0][0][2], corners[0][0][3])]
        self.mid_marker3 = [int(p) for p in midpoint(corners[0][0][3], corners[0][0][0])]

        cv2.circle(self.frame, self.mid_marker1, 2, (255, 0, 0), -1)
        cv2.circle(self.frame, self.mid_marker2, 2, (255, 0, 0), -1)
        cv2.circle(self.frame, self.mid_marker3, 2, (255, 0, 0), -1)
        cv2.circle(self.frame, self.mid_marker0, 2, (255, 0, 0), -1)
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
    def __init__(self, contour, ratio):
        self.contour = contour
        self.ratio = ratio

    def get_size(self):
        '''
        Returns size of sample.
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

        size1 = dist_0*self.ratio*1000
        size2 = dist_1*self.ratio*1000
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

        #cv2.rectangle(frame, (self.midpoints[0][0], self.midpoints[0][1]-8), (self.midpoints[0][0]+300, self.midpoints[0][1]-30), (0, 0, 0), -1)
        #cv2.putText(frame, f'Size: {size[0]:.1f}, {size[1]:.1f} Error: {error:.1f}', (self.midpoints[0][0], self.midpoints[0][1]-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
        cv2.circle(frame, self.midpoints[0], 3, (0, 255, 0), -1)
        cv2.circle(frame, self.midpoints[1], 3, (0, 255, 0), -1)
        cv2.circle(frame, self.midpoints[2], 3, (0, 255, 0), -1)
        cv2.circle(frame, self.midpoints[3], 3, (0, 255, 0), -1)

        cv2.drawContours(frame, [self.box.astype("int")], -1, (255, 255, 0), 1)

class VideoThread(QThread): 
    changePixmap = pyqtSignal(QImage)
    
    def __init__(self, camera_index=0, parent=None): 
        super().__init__(parent)
        self.laser_point = (320, 240)
        self._run_flag = True
        self.camera_index = camera_index
        self.taking_samples = False
    
    def save_image(self):
        return self.frame
    
    def set_gap_mm(self, gap):
        self.gap_mm = gap

    def show_laser(self):
        self.taking_samples = True

    # def run(self):
        #cap = cv2.VideoCapture(self.camera_index)
        # while self._run_flag:
            
        #     ret, bgrImage = cap.read()
        #     if not ret:
        #         print("Can't receive frame (stream end?). Exiting ...")
        #         break
           
        #     frame = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2RGB)
            
        #     h, w, ch = frame.shape
        #     if(self.taking_samples):
        #         cv2.circle(frame, (int(w/2), int(h/2)), 3, (255, 0, 255), -1)
        #     bytesPerLine = ch * w
        #     convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
        #     p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
        #     self.latest_frame = frame.copy()
        #     self.changePixmap.emit(p)
        # cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class ArduinoCamApp(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("design.ui", self)
        
        self.setStyleSheet(style_sheet)
        self.setWindowTitle("GUI")
        self.serial = QSerialPort(self) # Make serial part of the class
        self.serial.setBaudRate(115200)
        self.populate_ports()
        self.gap_mm = 1.0  # default gap

        ###
        self.toggle = AnimatedToggle(checked_color="#eb2149",pulse_checked_color="#5090eb")
        layout = QtWidgets.QVBoxLayout(self.modeSwitch)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle)
        self.toggle.stateChanged.connect(self.toggle_mode)
        ###

        self.stopButton.clicked.connect(self.stop)
        self.pauseButton.clicked.connect(self.pause)
        self.resumeButton.clicked.connect(self.resume)
        self.serial.readyRead.connect(self.onRead)
        self.openButton.clicked.connect(self.onOpen)
        self.closeButton.clicked.connect(self.onClose)
        self.closeButton.setEnabled(False)
        
        self.test_coordinates = []
        self.gapInputLineEdit.textChanged.connect(self.update_gap_mm)
        self.takePicButton.clicked.connect(self.takeImage)
        self.approveSampleButton.clicked.connect(self.giveCoords)
        self.homeButton.clicked.connect(self.homeBed)

        for i in range(-10000, 10000, 1000):
            for j in range(-50000, 50000, 5000):
                coord = (i, j)
                self.test_coordinates.append(coord)

        self.arduino_auto_mode = False
        self._serial_buffer = b''
        self.coord_index = 0
        self.sending_coords = False
        self.expected_ack = ""
        self.coord_reached = True
        self.home_position = (371, 392)
        self.laser_point = (240, 320)
        self.steps_per_mm_X = 10000/62.69
        self.steps_per_mm_Y = 30000/37.56
        
        self.video_thread = VideoThread(camera_index=4, parent=self)
        self.video_thread.changePixmap.connect(self.setImage)
        self.video_thread.start()
        
        self.showMaximized()

    def update_gap_mm(self):
        try:
            gap_value = float(self.gapInputLineEdit.text())
            self.gap_mm = gap_value
        except ValueError:
            pass  # You may want to show an error or fallback
    
    def takeSamples(self):
        if self.coordinates is not None:
            for c in self.coordinates:
                sample = ((c[0]-self.laser_point[0])*1000*self.ratio, (c[1]-self.laser_point[1])*1000*self.ratio)
                steps_x = -int(sample[1]*self.steps_per_mm_X)
                steps_y = int(sample[0]*self.steps_per_mm_Y)
                point = (steps_x, steps_y)
                #self.test_coordinates.append(point)

    def homeBed(self):
        if hasattr(self.video_thread, 'latest_frame'):
            frame = self.video_thread.latest_frame
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = detector.detectMarkers(frame_gray)
            h, w, _ = frame.shape

            if ids is not None:
                ids = ids.flatten()
                marker = Marker(corners, size=ARUCO_MARKER_SIZE, frame=frame)
                self.ratio = marker.get_ratio()
                marker_center = marker.get_center()
                difference_mm = ((self.home_position[0]-marker_center[0])*1000*self.ratio, (self.home_position[1]-marker_center[1])*1000*self.ratio)

            steps_x = int(difference_mm[0]*self.steps_per_mm_X)
            steps_y = -int(difference_mm[1]*self.steps_per_mm_Y)
            print("Homing...")
            self.sendModeCommand('h')

            if self.serial.isOpen():
                coord_str = f"X{steps_x} Y{steps_y}\n"
                self.serial.write(coord_str.encode('ASCII'))
                time.sleep(0.1)
            else:
                print("Serial port not open.")

    def takeImage(self):
    # Assuming you want the latest frame from the VideoThread
        if hasattr(self.video_thread, 'latest_frame'):
            frame = self.video_thread.latest_frame
            # Convert NumPy array to QImage
                
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_blur = cv2.bilateralFilter(frame_gray, 9, 40, 40)

            frame_show = frame.copy()
            #get corners of ArUco
            corners, ids, rejectedImgPoints = detector.detectMarkers(frame_gray)
            h, w, _ = frame.shape

            if ids is not None:
                ids = ids.flatten()
                marker = Marker(corners, size=ARUCO_MARKER_SIZE, frame=frame)
                self.ratio = marker.get_ratio()

            plate = cv2.adaptiveThreshold(
            frame_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 111, 15)

            try: 
                frame_lines, self.frame_vertices = prep_frame_lines(plate, frame_blur=frame_blur, frame_gray=frame_gray)
                lines = get_lines(frame_lines)
                (x_box, y_box, w_box, h_box) = self.frame_vertices
                if lines.any():
                    try:
                        contours, hierarchy = cv2.findContours(lines, cv2.RETR_TREE ,cv2.CHAIN_APPROX_NONE)
                        if contours:
                            #Filter out contours too large/too small
                            contours_filtered = [c for c in contours if 500<cv2.contourArea(c)<15000]
                            #Filter contours by size
                            contours_sorted = sorted(contours_filtered, key = cv2.contourArea, reverse=True)
                            #Grab the first (largest) contour
                            self.c = contours_sorted[0]
 
                            #Create sample object based on contour
                            sample = Sample(self.c, ratio=self.ratio)

                            #Measure size of object in mm
                            self.size = sample.get_size()
                            self.size_x.setText(str(round(self.size[0], 1)))
                            self.size_y.setText(str(round(self.size[1], 1)))

                            #Draw bounding rectangle of object
                            sample.draw_cont(real_size=(41.6, 48), frame = frame)

                            #Crop frame to only show bed with sample
                            frame_show = frame[y_box:y_box+h_box, x_box:x_box+w_box]
                            frame_show = cv2.resize(frame_show, (640, 480))
                    except Exception as e:
                        print('cont', e)
                        pass

            except Exception as e:
                print(e)
                pass
            
            #Process frame so it can be shown in GUI
            h_show, w_show, ch_show = frame_show.shape
            bytesPerLine = w_show * ch_show
            qt_image = QImage(frame_show.data, w_show, h_show, bytesPerLine, QImage.Format_RGB888)
            qt_image = qt_image.scaled(640, 480, Qt.KeepAspectRatio)
            self.sample.setPixmap(QPixmap.fromImage(qt_image))
            cv2.imwrite("sample_pic.jpg", frame)

    def giveCoords(self):
        frame = cv2.imread('sample_pic.jpg')
        h, w, ch = frame.shape
        #Create black mask to match size of image
        mask_black_cont = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_black_cont, [self.c], -1, color = 1, thickness=-1)
        (x_box, y_box, w_box, h_box) = self.frame_vertices

        if self.gap_mm <= 0:
            raise ValueError("gap cannot be 0 or less, choose continuous scanning instead")
        
        gap_pix = int(self.gap_mm/1000/self.ratio)

        mask_scan = np.zeros((h, w), dtype=np.uint8)

        for i in range(0, h, gap_pix):
            for j in range(0, w, gap_pix):
                index = (i, j)
                if mask_black_cont[index]:
                    mask_scan[index] = 1                    

        y_shape, x_shape = np.where(mask_scan == 1)
        contour_ones = list(zip(y_shape, x_shape))     
        
        contour_ones = ([(int(c[0]), int(c[1])) for c in contour_ones])
        
        for i in contour_ones:
            frame[i] = (255, 0, 0)
        
        frame_show = frame[y_box:y_box+h_box, x_box:x_box+w_box]

        #Convert NumPy array to QImage
        frame_show = cv2.resize(frame_show, (640, 480))
        h_show, w_show, ch_show = frame_show.shape
        
        bytesPerLine = w_show * ch_show
        self.sampleCoordLabel.setText(str(contour_ones))
        self.coordinates = contour_ones
        print(self.coordinates)
        qt_image = QImage(frame_show.data, w_show, h_show, bytesPerLine, QImage.Format_RGB888)
        qt_image = qt_image.scaled(640, 480, Qt.KeepAspectRatio)
        self.sample.setPixmap(QPixmap.fromImage(qt_image))
        self.takeSamples()
        #self.yCoordLabel.setText(coords)

    def populate_ports(self): 
        self.comL.clear() # Clear existing items
        portList = []
        ports = QSerialPortInfo.availablePorts()
        for port in ports:
            portList.append(port.portName())
        self.comL.addItems(portList)

    def setImage(self, image):
        self.video.setPixmap(QPixmap.fromImage(image))

    def onRead(self):
        while self.serial.canReadLine():
            rx_bytearray = self.serial.readLine()
            self._serial_buffer += rx_bytearray
            while b'\n' in self._serial_buffer:
                line, self._serial_buffer = self._serial_buffer.split(b'\n')
                rxs = bytes(line).decode('utf-8', errors='ignore').strip()
                print(f"Received: {rxs}")  # Debug print

                if rxs == "auto":
                    self.arduino_auto_mode = True
                    print("Arduino: AUTO mode active.")
                elif rxs == "manual":
                    self.arduino_auto_mode = False
                    print("Arduino: MANUAL mode active.")
                elif rxs == "R":
                    self.coord_reached = True
                    time.sleep(0.2)

                    cap = picam2.capture_array()
                    ret, bgrImage = cap.read()
                    
                    if not ret:
                        print("Can't receive frame (stream end?). Exiting ...")
                        break
            
                    frame = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(f"/home/david/Desktop/GUI/test_images_raspi/test_pic{self.coord_index}.jpg",frame )
                    
                    if self.coord_index<len(self.test_coordinates):
                        self.sendCoordinates(self.test_coordinates)

                # else:
                #     print(f"Unexpected response: {rxs}")

    def onOpen(self):
         port_name = self.comL.currentText()
         self.serial.setPortName(port_name) 
         self.serial.open(QIODevice.ReadWrite)

    def onClose(self):
        self.serial.close()

    def toggle_mode(self, state):
        if state:
            print("Auto mode selected")
            self.sendModeCommand('a')
            time.sleep(0.2)
            self.arduino_auto_mode = True
            self.coord_index = 0
            self.video_thread.show_laser()
            self.sendCoordinates(self.test_coordinates)
        
        else:
            print("Manual mode selected")
            self.sendModeCommand('m')
            self.arduino_auto_mode = False
            time.sleep(0.2)
    
    def sendModeCommand(self, command_char):
        if self.serial.isOpen():
            self.serial.write((command_char + '\n').encode('ASCII'))
            print(f"Sent: {command_char}")
            time.sleep(0.1)  # Add a short delay to allow Arduino to process the command
        else:
            print("Serial port not open.")
    
    def sendCoordinates(self, coordinates):
        coord = coordinates[self.coord_index]
        if self.coord_reached:
            if self.serial.isOpen():
                coord_str = f"X{coord[0]} Y{coord[1]}\n"
                self.serial.write(coord_str.encode('ASCII'))
                print(f"Sent: {coord_str.strip()}, coord num: {self.coord_index}")
                time.sleep(0.1)
            else:
                print("Serial port not open.")
            self.coord_index+=1
            self.coord_reached = False
        elif self.coord_index == len(self.test_coordinates):
            taking_samples = False
            self.coord_index = 0    

    def stop(self):
        self._running = False
        
        if self.serial.isOpen():
            self.sendCommand("c")
            print("Stop command sent to Arduino.")
        else:
            print("Serial port not open.")

    def pause(self):
        
        if self.serial.isOpen():
            self.sendCommand("v")
            print("Pause command sent to Arduino.")
        else:
            print("Serial port not open.")

    def resume(self):
        
        if self.serial.isOpen():
            self.sendCommand("b")
            print("Resume command sent to Arduino.")
        else:
            print("Serial port not open.")


time.sleep(1) 

if __name__ == '__main__':
    app = QApplication(sys.argv) 
    mainWindow = ArduinoCamApp() 
    sys.exit(app.exec_())
