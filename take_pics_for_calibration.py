import numpy as np
import cv2 as cv

from timeit import default_timer as timer

prev_time = 0
time_start = int(timer())

cap = cv.VideoCapture(4)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    
    time_elapsed = int(timer()) - time_start

    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame, exit")
        break
    
    print(time_elapsed)
    if time_elapsed != prev_time:
        cv.imwrite(f"/home/david/Desktop/ComputerVision/calib_pics/pic{time_elapsed}.jpg", frame)
        print("time saved")
        prev_time = time_elapsed
    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()