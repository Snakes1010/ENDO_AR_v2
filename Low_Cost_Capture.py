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

chessboardSize = (10,7)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (0,1,0), (0,2,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
xv, yv = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]]
objp[:, :2] = np.array([yv,xv]).T.reshape(-1,2)
objpoints = []
imgpoints_L = [] # 2d points in image plane.
imgpoints_R = [] # 2d points in image plane.

save_dir_L = "/home/jacob/endo_calib/low_cost_proj/8_11_2x/stereoframesL"
save_dir_R = "/home/jacob/endo_calib/low_cost_proj/8_11_2x/stereoframesR"
os.makedirs(save_dir_L, exist_ok=True)
os.makedirs(save_dir_R, exist_ok=True)

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
    frame_L_saved = frame_L.copy()
    frame_R_saved = frame_R.copy()
    gray_img_L = cv.cvtColor(frame_L, cv.COLOR_BGR2GRAY)
    gray_img_R = cv.cvtColor(frame_R, cv.COLOR_BGR2GRAY)
    ret_cb_L, cornersL = cv.findChessboardCornersSB(gray_img_L, chessboardSize, flags=None)
    ret_cb_R, cornersR = cv.findChessboardCornersSB(gray_img_R, chessboardSize, flags=None)
    combined_gray = np.concatenate((gray_img_L, gray_img_R), axis=1)

    if ret_cb_L == True and ret_cb_R == True:
        objpoints.append(objp)
        imgpoints_L.append(cornersL)
        imgpoints_R.append(cornersR)
        cv.drawChessboardCorners(frame_L, chessboardSize, cornersL, ret_cb_L)
        cv.drawChessboardCorners(frame_R, chessboardSize, cornersR, ret_cb_R)
        combined = np.concatenate((frame_L, frame_R), axis=1)
        cv.imshow('combined', combined)
    else:
        cv.imshow('combined', combined_gray)
    if cv.waitKey(1) & 0xFF == ord('s'):
        filepath_L = os.path.join(save_dir_L, f"frame_{frame_counter}.png")
        filepath_R = os.path.join(save_dir_R, f"frame_{frame_counter}.png")
        cv.imwrite(filepath_L, frame_L_saved)
        cv.imwrite(filepath_R, frame_R_saved)
        print(f"Saved frame {frame_counter}")
        frame_counter += 1
    if cv.waitKey(1) & 0xFF == ord('q'):
            break

left_cam.release()
right_cam.release()
cv.destroyAllWindows()
