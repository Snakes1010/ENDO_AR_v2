import cv2 as cv
import numpy as np
import os
# left cam = 0 rightcam =2
cap = cv.VideoCapture(0)
right_cam = cv.VideoCapture(2)

print_shape_once = True

cap.set(cv.CAP_PROP_BRIGHTNESS, 0)  # Brightness: -64 to 64, default: 0
cap.set(cv.CAP_PROP_CONTRAST, 32)  # Contrast: 0 to 64, default: 32
cap.set(cv.CAP_PROP_SATURATION, 64)  # Saturation: 0 to 128, default: 64
cap.set(cv.CAP_PROP_HUE, 0)  # Hue: -40 to 40, default: 0
cap.set(cv.CAP_PROP_AUTO_WB, 1)  # White Balance, Automatic: 0 or 1, default: 1
cap.set(cv.CAP_PROP_GAMMA, 100)  # Gamma: 72 to 500, default: 100
cap.set(cv.CAP_PROP_GAIN, 0)  # Gain: 0 to 100, default: 0
cap.set(cv.CAP_PROP_SHARPNESS, 2)  # Sharpness: 0 to 6, default: 2
cap.set(cv.CAP_PROP_BACKLIGHT, 1)  # Backlight Compensation: 0 to 4, default: 1
cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)  # Auto Exposure: 0 to 3, default: 1
cap.set(cv.CAP_PROP_EXPOSURE, 250)  # Exposure Time, Absolute: 1 to 5000, default: 473
cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0)  # Exposure, Dynamic Framerate: 0 or 1, default: 0
# width, height = 1280, 720
# cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)


save_dir = "/home/jacob/endo_calib/low_cost_proj/8_11_2x/7_25_frames_L_auto_off"
os.makedirs(save_dir, exist_ok=True)


frame_counter = 0

while True:
    ret_L, frame_L = cap.read()
    _,_ = right_cam.read()

    if not ret_L:
        print('ERROR: Could not open stereo rig')
        break
    else:
        cv.imshow('image', frame_L)
    if cv.waitKey(1) & 0xFF == ord('s'):
        filepath_L = os.path.join(save_dir, f"frame_{frame_counter}.png")
        cv.imwrite(filepath_L, frame_L)
        print(f"Saved frame {frame_counter}")
        frame_counter += 1
    if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
