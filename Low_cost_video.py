import cv2 as cv
import numpy as np
import os

left_cam = cv.VideoCapture(0)
right_cam = cv.VideoCapture(2)

print_shape_once = True


properties = [cv.CAP_PROP_POS_MSEC, cv.CAP_PROP_POS_FRAMES, cv.CAP_PROP_POS_AVI_RATIO,
              cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT, cv.CAP_PROP_FPS,
              cv.CAP_PROP_FOURCC, cv.CAP_PROP_FRAME_COUNT, cv.CAP_PROP_FORMAT,
              cv.CAP_PROP_MODE, cv.CAP_PROP_BRIGHTNESS, cv.CAP_PROP_CONTRAST,
              cv.CAP_PROP_SATURATION, cv.CAP_PROP_HUE, cv.CAP_PROP_GAIN,
              cv.CAP_PROP_EXPOSURE]

# Property names
property_names = ["POS_MSEC", "POS_FRAMES", "POS_AVI_RATIO", "FRAME_WIDTH",
                  "FRAME_HEIGHT", "FPS", "FOURCC", "FRAME_COUNT", "FORMAT", "MODE",
                  "BRIGHTNESS", "CONTRAST", "SATURATION", "HUE", "GAIN", "EXPOSURE"]

save_dir_L = "/home/jacob/endo_calib/low_cost_proj/8_11_2x/stereo_video_L.mp4"
save_dir_R = "/home/jacob/endo_calib/low_cost_proj/8_11_2x/stereo_video_R.mp4"


frame_size = (640,480)
# Define the codec and create a VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out_L = cv.VideoWriter(save_dir_L, fourcc, 20.0, frame_size)
out_R = cv.VideoWriter(save_dir_R, fourcc, 20.0, frame_size)
frame_counter = 0

for prop in properties:
    valueL = left_cam.get(prop)
    valueR = right_cam.get(prop)
    successL = left_cam.set(prop, valueL)
    successR = right_cam.set(prop, valueR)
    if successL and successR:
        print(f"Property Left {prop} set to {valueL}")
        print(f"Property Right {prop} set to {valueR}")
    else:
        print(f"Failed to set property {prop}")

while True:
    ret_L, frame_L = left_cam.read()
    ret_R, frame_R = right_cam.read()
    if not ret_L or not ret_R:
        print('ERROR: Could not open stereo rig')
        break


    combined = np.concatenate((frame_L, frame_R), axis=1)
    out_L.write(frame_L)
    out_R.write(frame_R)
    cv.imshow('combined', combined)
    if cv.waitKey(1) & 0xFF == ord('q'):
            break

left_cam.release()
right_cam.release()
cv.destroyAllWindows()
