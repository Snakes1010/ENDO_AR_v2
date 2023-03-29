# This program imports a folder containing frames of a calibration video
# First is employs a trackbar and the CB_fast algo to check which frames CB is detedted
# Clicking 's' saves the matched frames to a list that will undergo fine CB detection
# Clickign 'q' will stop the while loops and if the frame is empyty terminate the program
# If the list contatins images when the 'q' is pressed calibration occurs

import cv2 as cv
import glob
import os
import numpy as np

# identify the location of where we want to save the quality frames
dir_name = "/Users/jcsimon/Desktop/ENDO_AR/data_2_3_23"
################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################
# chessboardSize = (col, row)
chessboardSize = (10,7)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (0,1,0), (0,2,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
xv, yv = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]]
objp[:, :2] = np.array([yv,xv]).T.reshape(-1,2)
scalefactor = 1
imgloop = 0
# Arrays to store object points and image points from all the images.
objpoints_fast = [] # 3d point in real world space
imgpointsL_fast = [] # 2d points in image plane.
imgpointsR_fast = [] # 2d points in image plane.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.
matched_L = []
matched_R = []


# if not os.path.exists(dir_name):
#     os.makedirs(dir_name)
# grab extracted frames from a folder and add it to a list and sort it
images_left = glob.glob('/Users/jcsimon/Desktop/ENDO_AR/data_2_3_23/frameL/*')
images_right = glob.glob('/Users/jcsimon/Desktop/ENDO_AR/data_2_3_23/frameR/*')
images_left_sort = sorted(images_left)
images_right_sort = sorted(images_right)
left_chosen = []
left_chosen_path = []
right_chosen =[]
right_chosen_path =[]
dimensions = cv.imread(images_left_sort[0])
dimensions = dimensions.shape
print("FRAME DIMENSIONS:\n", dimensions)
# initialize a counter to frame = 0
currentframe = 0

def on_trackbar(pos):
    pass

# starts the display window event that relies on the function on_trackbar
cv.namedWindow('combined')
cv.createTrackbar('Frame', 'combined', 0, len(images_left_sort) - 1, on_trackbar)

while True:
    pos = cv.getTrackbarPos('Frame', 'combined')
    img_l_fast = cv.imread(images_left_sort[pos])
    img_r_fast = cv.imread(images_right_sort[pos])
    grayL_fast = cv.cvtColor(img_l_fast, cv.COLOR_BGR2GRAY)
    grayR_fast = cv.cvtColor(img_r_fast, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    retL, cornersL_fast = cv.findChessboardCorners(grayL_fast, chessboardSize, None, flags=cv.CALIB_CB_FAST_CHECK)
    retR, cornersR_fast = cv.findChessboardCorners(grayR_fast, chessboardSize, None, flags=cv.CALIB_CB_FAST_CHECK)
    # If found, add object points, image points (after refining them)
    if retL and retR == True:
        objpoints_fast.append(objp)
        cornersL_fast = cv.cornerSubPix(grayL_fast, cornersL_fast, (11, 11), (-1, -1), criteria)
        imgpointsL_fast.append(cornersL_fast)
        cornersR_fast = cv.cornerSubPix(grayR_fast, cornersR_fast, (11, 11), (-1, -1), criteria)
        imgpointsR_fast.append(cornersR_fast)
        # Draw and display the corners
        cv.drawChessboardCorners(img_l_fast, chessboardSize, cornersL_fast, retL)
        cv.drawChessboardCorners(img_r_fast, chessboardSize, cornersR_fast, retR)
        combined = np.concatenate((img_l_fast, img_r_fast), axis=1)
        cv.putText(combined, str(pos), (10, 50), cv.FONT_ITALIC, 2, (255, 255, 255), 2, cv.LINE_AA)
    else:
        combined = np.concatenate((img_l_fast, img_r_fast), axis=1)
        cv.putText(combined, "NO MATCH FOUND_" + str(pos), (10, 50), cv.FONT_ITALIC, 2, (0, 0, 255), 4, cv.LINE_AA)
    cv.imshow("combined", combined)
    key = cv.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        left_chosen.append(img_l_fast)
        left_chosen_path.append(images_left_sort[pos])
        right_chosen.append(img_r_fast)
        right_chosen_path.append(images_right_sort[pos])
        print(left_chosen_path)
        print(right_chosen_path)
        print("Image Added #_" + str(len(left_chosen)))

cv.destroyAllWindows()

if len(left_chosen_path) != 0 or len(right_chosen_path) != 0:
    for img_l, img_r in zip(left_chosen_path, right_chosen_path):
        imgloop = imgloop + 1
        imgloopstr = str(imgloop)
        img_l = cv.imread(img_l)
        img_r = cv.imread(img_r)
        grayL_chosen = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
        grayR_chosen = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        retL_chosen, cornersL_chosen = cv.findChessboardCorners(grayL_chosen, chessboardSize, None)
        retR_chosen, cornersR_chosen = cv.findChessboardCorners(grayR_chosen, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if retL_chosen and retR_chosen == True:

            objpoints.append(objp)
            # R and L subpixel
            cornersL = cv.cornerSubPix(grayL_chosen, cornersL_chosen, (11,11), (-1,-1), criteria)
            imgpointsL.append(cornersL)
            cornersR = cv.cornerSubPix(grayR_chosen, cornersR_chosen, (11,11), (-1,-1), criteria)
            imgpointsR.append(cornersR)

            # Draw and display the corners
            cv.drawChessboardCorners(img_l, chessboardSize, cornersL, retL_chosen)
            cv.drawChessboardCorners(img_r, chessboardSize, cornersR, retR_chosen)

            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(img_l, imgloopstr, (50,100),font,1, (0,0,255), 2)
            cv.putText(img_r, imgloopstr, (50, 100), font, 1, (0, 0, 255), 2)
            combined = np.concatenate((img_l, img_r), axis=1)
            cv.imshow('combined', combined)
            cv.waitKey(1000)

    cv.destroyAllWindows()

    print(imgpointsL)
    ############## CALIBRATION #######################################################
    #imageSize Size of the image used only to initialize the intrinsic camera matrix [w,h].
    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, (dimensions[1], dimensions[0]), None, None)
    newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (dimensions[1], dimensions[0]), 1, (dimensions[1], dimensions[0]))

    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, (dimensions[1], dimensions[0]), None, None)
    newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (dimensions[1], dimensions[0]), 1, (dimensions[1], dimensions[0]))
    print('rmse_l: ', retL)
    print('rmse_r: ', retR)
    print('newCameraMatrixL:\n', newCameraMatrixL)
    print('newCameraMatrixR:\n', newCameraMatrixR)

     ########## Stereo Vision Calibration #############################################
    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC
    # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
    # Hence intrinsic parameters are the same
    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # This step is performed to transformation between the two cameras and calculate Essential and Fundamental matrix
    # Size of the image used only to initialize the intrinsic camera matrix [w,h].
    retStereo, StereoCameraMatrixL, distL_stereo, StereoCameraMatrixR, distR_stereo, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, (dimensions[1], dimensions[0]), criteria_stereo, flags)
    print('rot',rot)

    # StereoVison rectification
    rectifyScale = 1
    rectL_stereo, rectR_stereo, projMatrixL, projMatrixR, Q, roi_L2, roi_R2 = cv.stereoRectify(StereoCameraMatrixL, distL_stereo, StereoCameraMatrixR, distR_stereo, (dimensions[1], dimensions[0]), rot, trans, rectifyScale,(0,0))
    print('Q', Q)
    print('projMatrixL', projMatrixR)
    print(rectR_stereo)
    print(rectL_stereo)
    #
    stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL_stereo, rectL_stereo, projMatrixL, (dimensions[1], dimensions[0]), cv.CV_16SC2)
    stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR_stereo, rectR_stereo, projMatrixR, (dimensions[1], dimensions[0]), cv.CV_16SC2)

    print("Saving parameters!")
    cv_file = cv.FileStorage('stereoMap2_24ksh_23.xml', cv.FILE_STORAGE_WRITE)

    cv_file.write('stereoMapL_x',stereoMapL[0])
    cv_file.write('stereoMapL_y',stereoMapL[1])
    cv_file.write('stereoMapR_x',stereoMapR[0])
    cv_file.write('stereoMapR_y',stereoMapR[1])
