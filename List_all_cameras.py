import cv2
import os

print('CPU', os.name)
# Loop through camera indexes until a camera cannot be opened
index = 0
while True:
    print(index)
    cap = cv2.VideoCapture(index)
    connected = cap.isOpened()
    if connected:
        print(f"Camera {index} connected")
        cap.release()
    index += 1
    if index >= 9:
        print('done')
        break
cap.release()
