import time
import numpy as np
import cv2 as cv

from picamera2 import Picamera2, Preview
from libcamera import controls, Transform, ColorSpace

img_width = 640
img_height = 480

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration( transform = Transform(hflip=1, vflip=1), colour_space=ColorSpace.Sycc()))
picam2.configure(picam2.create_video_configuration(main = {"size": (img_width, img_height)}))
picam2.start()
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
print(picam2.camera_properties)

time.sleep(2)
track_pos = 100

def on_change(track_pos):
	picam2.set_controls({"ScalerCrop": (x_offset, y_offset, width, height)} )		
cv.namedWindow('image')


cv.createTrackbar('Z', 'image', 1, 14, on_change)
cv.createTrackbar('X', 'image', 0, 3000, on_change)
cv.createTrackbar('Y', 'image', 0, 3000, on_change)

while True:
#    print(picam2.camera_controls['ScalerCrop'])
    frame = picam2.capture_array()

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    x_offset = cv.getTrackbarPos('X', 'image')
    y_offset = cv.getTrackbarPos('Y', 'image')
    zoom_factor = cv.getTrackbarPos('Z', 'image') if cv.getTrackbarPos('Z', 'image') > 0 else 1   
    width = 3072 // zoom_factor
    height = 1728 // zoom_factor

    #combined_image = cv.hconcat([frame, image])
    
    cv.imshow('frame', frame)
    #cv.imshow('combined', combined_image)
    
    if cv.waitKey(1) == ord('q'):
        cv.imwrite("/home/pi/Desktop/aie2-1/pics/frame.jpg", frame)
        break

time.sleep(2)
