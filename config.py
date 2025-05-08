import cv2
import numpy as np

# Calibration parameters yaml file
CAMERA_CALIBRATION_PARAMETERS_FILENAME = 'calibration_chessboard.yaml'

#ARUCO dictionary used for marker detection
ARUCO_DICT_CONFIG = cv2.aruco.DICT_4X4_50

ARUCO_MARKER_SIZE = 0.025

style_sheet = """

QMainWindow, QWidget {
    background-color: #f0f0f0; 
}

QLabel {
    font-size: 10pt;
    color: #333333; 
    padding: 2px; 
}


#xCoordLabel, #yCoordLabel, #speedLabel {
    font-weight: bold;
    font-size: 11pt;
    color: #2E8B57; 
    padding: 3px 5px;
}

#statusLabel {
    font-size: 9pt;
    font-style: italic;
    color: #555555; 
    padding: 3px 5px;
}

#video {
    background-color: black; 
    border: 2px solid #aaaaaa; 
}

QLineEdit {
    background-color: white;
    border: 1px solid #aaaaaa;
    border-radius: 3px;
    padding: 4px 6px; 
    font-size: 10pt;
    color: #222222;
}
QLineEdit:focus {
    border: 1px solid #4682B4; 
}
QLineEdit:disabled {
    background-color: #e8e8e8; 
    color: #888888;
}



QPushButton {
    background-color: #e0e0e0; 
    color: #111111; 
    border: 1px solid #bdbdbd;
    border-radius: 4px;
    padding: 5px 10px;
    font-size: 9pt;
    min-width: 60px; 
}
QPushButton:hover {
    background-color: #eeeeee; 
    border: 1px solid #ababab;
}
QPushButton:pressed {
    background-color: #d0d0d0; 
    padding: 6px 9px 4px 11px; 
}
QPushButton:disabled {
    background-color: #f5f5f5; 
    color: #aaaaaa; 
    border: 1px solid #e0e0e0;
}


#openButton, #closeButton, #getPosButton {
    background-color: #4682B4; 
    color: white;
    font-weight: bold;
    border: 1px solid #386890;
    min-width: 70px;
}
#openButton:hover, #closeButton:hover, #getPosButton:hover {
    background-color: #5A9BD8;
    border: 1px solid #4682B4;
}
#openButton:pressed, #closeButton:pressed, #getPosButton:pressed {
    background-color: #36648B;
}

#openButton:disabled, #closeButton:disabled, #getPosButton:disabled {
     background-color: #a0b4c7; 
     color: #e0e0e0;
     border: 1px solid #8ba0b3;
}



#sendXButton, #sendYButton, #motorSpeedButton {
    background-color: #5cb85c; 
    color: white;
    font-weight: bold;
    border: 1px solid #4cae4c;
    min-width: 65px;
}
#sendXButton:hover, #sendYButton:hover, #motorSpeedButton:hover {
    background-color: #6cd96c;
    border: 1px solid #5cb85c;
}
#sendXButton:pressed, #sendYButton:pressed, #motorSpeedButton:pressed {
    background-color: #458d45;
}
#sendXButton:disabled, #sendYButton:disabled, #motorSpeedButton:disabled {
    background-color: #a8d3a8; 
    color: #e8f5e8;
    border: 1px solid #97bf97;
}

#comL {
    background-color: white;
    border: 1px solid #aaaaaa;
    border-radius: 3px;
    padding: 3px 5px; 
    min-width: 6em; 
}
#comL:focus {
    border: 1px solid #4682B4; 
}

#comL::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 15px;
    border-left-width: 1px;
    border-left-color: #aaaaaa;
    border-left-style: solid;
    border-top-right-radius: 3px;
    border-bottom-right-radius: 3px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f6f6f6, stop:1 #e0e0e0);
}
#comL::down-arrow {
    image: url('images/arrow.png'); 
    width: 10px; 
    height: 10px;
}
#comL::down-arrow:on { 
    top: 1px;
    left: 1px;
}

#comL QAbstractItemView {
    border: 1px solid #999999;
    background-color: white;
    selection-background-color: #4682B4; 
    color: #333333;
}
#comL:disabled {
    background-color: #e8e8e8;
    color: #888888;
    border: 1px solid #cccccc;
}
#comL::drop-down:disabled {
     background: #e8e8e8;
     border-left-color: #cccccc;
}

#progressBar {
    border: 1px solid #bdbdbd;
    border-radius: 4px;
    background-color: #e0e0e0; 
    text-align: center; 
    color: #555555; 
}
#progressBar::chunk {
    background-color: #4CAF50; 
    border-radius: 3px; 
    
}
"""