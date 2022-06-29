import numpy as np
import cv2 
 
cap = cv2.VideoCapture('./test.mp4')

cnt = 0
while True:
    ret, frame = cap.read() 
    cv2.imwrite('./frames/' + str(cnt) + '.jpg', frame)
    cnt += 1
    if not ret:
        break
