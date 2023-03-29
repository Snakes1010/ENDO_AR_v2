# This program imports of matched pair of frames that should have been chosen from the fast
# CB algo, 7-15 frames in total, and runs the fine CB detection and calibrates stereo

import numpy as np
import cv2 as cv
import glob
import time

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################
chessboardSize = (10,7)
Scalefactor = 1
frameSize = (int(1920/Scalefactor), int(1080/Scalefactor))
print(frameSize)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (0,1,0), (0,2,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
xv, yv = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]]
objp[:, :2] = np.array([yv,xv]).T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.
matched_L = []
matched_R = []
imgloop = 0

imagesLeft = glob.glob('/Users/jcsimon/Desktop/ENDO_AR/data_2_3_23/frameL/*')
imagesRight = glob.glob('/Users/jcsimon/Desktop/ENDO_AR/data_2_3_23/frameR/*')
imagesLeft_sort = sorted(imagesLeft)
imagesRight_sort = sorted(imagesRight)

## SCAN video and find Matching CBs to then run in detail
for imgLeft, imgRight in zip(imagesLeft_sort, imagesRight_sort):
    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None, flags=cv.CALIB_CB_FAST_CHECK)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None, flags=cv.CALIB_CB_FAST_CHECK)

    # If found, add object points, image points (after refining them)
    if retL and retR == True:
        print('matchedpairs\n', imgLeft.strip('/Users/jcsimon/Desktop/ENDO_AR/'), imgRight.strip('/Users/jcsimon/Desktop/ENDO_AR/'))
        matched_L.append(imgLeft)
        matched_R.append(imgRight)

print(matched_L)
print(len(matched_L))

print(matched_R)
print(len(matched_R))

for imgLeft_cb, imgRight_cb in zip(matched_L, matched_R):
    imgleftstr_cb = str(imgLeft_cb)
    imgrightstr_cb = str(imgRight_cb)
    imgloop = imgloop + 1
    imgloopstr = str(imgloop)
    imgL_cb = cv.imread(imgLeft_cb)
    imgR_cb = cv.imread(imgRight_cb)
    grayL_cb = cv.cvtColor(imgL_cb, cv.COLOR_BGR2GRAY)
    grayR_cb = cv.cvtColor(imgR_cb, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    retL_cb, cornersL_cb = cv.findChessboardCorners(grayL_cb, chessboardSize, None)
    retR_cb, cornersR_cb = cv.findChessboardCorners(grayR_cb, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if retL_cb and retR_cb == True:

        objpoints.append(objp)

        # R and L subpixel
        cornersL_cb = cv.cornerSubPix(grayL_cb, cornersL_cb, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL_cb)
        cornersR_cb = cv.cornerSubPix(grayR_cb, cornersR_cb, (11,11), (-1,-1), criteria)
        imgpointsR.append(cornersR_cb)

        # Draw and display the corners
        cv.drawChessboardCorners(imgL_cb, chessboardSize, cornersL_cb, retL_cb)
        cv.drawChessboardCorners(imgR_cb, chessboardSize, cornersR_cb, retR_cb)

        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(imgL_cb,imgleftstr_cb, (50,50), font,1, (0,0,255), 2)
        cv.putText(imgL_cb, imgloopstr, (50,100),font,1, (0,0,255), 2)
        cv.putText(imgR_cb, imgrightstr_cb, (50, 50), font, 1, (0, 0, 255), 2)
        cv.putText(imgR_cb, imgloopstr, (50, 100), font, 1, (0, 0, 255), 2)
        combined = np.concatenate((imgL_cb, imgR_cb), axis=1)
        cv.imshow('combined', combined)
        cv.waitKey(1)

cv.destroyAllWindows()

# print(objpoints[0])
print(len(imgpointsR))
print(imgpointsR[0].shape)
print(imgpointsR[0].ndim)
############## CALIBRATION #######################################################
#imageSize Size of the image used only to initialize the intrinsic camera matrix [w,h].
retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
heightL, widthL = grayL.shape
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

print('graypaheL:\n',widthL)
retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
heightR, widthR = grayR.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

print('CameraMatrixL:\n', cameraMatrixL)
print('CameraMatrixR:\n', cameraMatrixR)
print('newCameraMatrixL:\n', newCameraMatrixL)
print('newCameraMatrixR:\n', newCameraMatrixR)
print('rvexL\n', len(rvecsR))
print('tvecs\n', len(tvecsR))
print('rvexL\n', rvecsR[0])
#
# #
# # ########## Stereo Vision Calibration #############################################
#
# flags = 0
# flags |= cv.CALIB_FIX_INTRINSIC
# # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# # Hence intrinsic parameters are the same
#
# criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
# # This step is performed to transformation between the two cameras and calculate Essential and Fundamental matrix
# # Size of the image used only to initialize the intrinsic camera matrix [w,h].
# retStereo, StereoCameraMatrixL, distL_stereo, StereoCameraMatrixR, distR_stereo, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, frameSize, criteria_stereo, flags)
#
# print('rot',rot)
#
# # StereoVison rectification
# rectifyScale = 1
# rectL_stereo, rectR_stereo, projMatrixL, projMatrixR, Q, roi_L2, roi_R2 = cv.stereoRectify(StereoCameraMatrixL, distL_stereo, StereoCameraMatrixR, distR_stereo, frameSize, rot, trans, rectifyScale,(0,0))
# print('Q', Q)
# print('projMatrixL', projMatrixR)
# print(rectR_stereo)
# print(rectL_stereo)
# #
# stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL_stereo, rectL_stereo, projMatrixL, frameSize, cv.CV_16SC2)
# stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR_stereo, rectR_stereo, projMatrixR, frameSize, cv.CV_16SC2)
#
# # objp2= np.ones((1080 , 1920), np.intc)
# # # print(stereoMapR)
# # # print(stereoMapL)
# # #
# print("Saving parameters!")
# cv_file = cv.FileStorage('stereoMap1_22_22.xml', cv.FILE_STORAGE_WRITE)
#
# cv_file.write('stereoMapL_x',stereoMapL[0])
# cv_file.write('stereoMapL_y',stereoMapL[1])
# cv_file.write('stereoMapR_x',stereoMapR[0])
# cv_file.write('stereoMapR_y',stereoMapR[1])
# # # #
#
# # # #
# # # # #CALUCALTE ERROR
# # # # mean_errorL = 0
# # # # mean_errorR = 0
# # # # for i in range(len(objpoints)):
# # # #     imgpoints2L, _ = cv.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], cameraMatrixL, distL)
# # # #     errorL = cv.norm(imgpointsL[i], imgpoints2L, cv.NORM_L2)/len(imgpoints2L)
# # # #     mean_errorL += errorL
# # # #     imgpoints2R, _ = cv.projectPoints(objpoints[i], rvecsR[i], tvecsR[i], cameraMatrixR, distR)
# # # #     errorR = cv.norm(imgpointsR[i], imgpoints2R, cv.NORM_L2) / len(imgpoints2R)
# # # #     mean_errorR += errorR
# # # #
# # # # print("\ntotal error Left: {}".format(mean_errorL/len(objpoints)))
# # # # print("\ntotal error Right: {}".format(mean_errorR/len(objpoints)))
# #################################