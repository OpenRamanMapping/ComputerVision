import time
import numpy as np
import cv2 as cv

from picamera2 import Picamera2, Preview
from libcamera import controls, Transform, ColorSpace

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration( transform = Transform(hflip=1, vflip=1), colour_space=ColorSpace.Sycc(), lores={"size": (320, 240)}, display="lores"))
#picam2.configure(picam2.create_video_configuration(main = {"size": (640, 480)})
picam2.start()
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
#print(picam2.camera_properties)

time.sleep(2)
track_pos = 100

def on_change(track_pos):
	picam2.set_controls({"ScalerCrop": (0, 0, track_pos, track_pos)} )
	
cv.namedWindow('image')
cv.createTrackbar('F', 'image', 0, 1728, on_change)

while True:
    
    print(picam2.camera_controls['ScalerCrop'])
    frame = picam2.capture_array()
    
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    focus = cv.getTrackbarPos('F', 'image')
    
    #combined_image = cv.hconcat([frame, image])
    
    cv.imshow('frame', frame)
    #cv.imshow('combined', combined_image)
    
    if cv.waitKey(1) == ord('q'):
        cv.imwrite("/home/pi/Desktop/aie2-1/pics/frame.jpg", frame)
        break

time.sleep(2)
