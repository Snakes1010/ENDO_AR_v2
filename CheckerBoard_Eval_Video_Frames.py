# This program opens a folder containing camera frames and employs the fast checkerboard algo
# it presents only the matched pairs where both detected the CB

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
matched_pairs =[]
imgloop = 0


imagesLeft = glob.glob('/Users/jcsimon/Desktop/ENDO_AR/data_2_3_23/frameL/*')
imagesRight = glob.glob('/Users/jcsimon/Desktop/ENDO_AR/data_2_3_23/frameR/*')
imagesLeft_sort = sorted(imagesLeft)
imagesRight_sort = sorted(imagesRight)


for imgLeft, imgRight in zip(imagesLeft_sort, imagesRight_sort):
    # print(imgloop)
    imgleftstr = str(imgLeft)
    imgrightstr = str(imgRight)
    imgloop = imgloop + 1
    imgloopstr = str(imgloop)
    # print('matchedpairs\n', imgLeft, imgRight)
    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    imgL = cv.resize(imgL, frameSize)
    imgR = cv.resize(imgR, frameSize)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None,flags=cv.CALIB_CB_FAST_CHECK)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None, flags=cv.CALIB_CB_FAST_CHECK)

    # If found, add object points, image points (after refining them)
    if retL and retR == True:
        print('matchedpairs\n', imgLeft.strip('/Users/jcsimon/Desktop/ENDO_AR/'), imgRight.strip('/Users/jcsimon/Desktop/ENDO_AR/'))
        matched_pairs.append(imgLeft.strip('/Users/jcsimon/Desktop/ENDO_AR/'))
        print(matched_pairs)
        print(imgloop)
        objpoints.append(objp)
        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(cornersR)

        # Draw and display the corners
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)

        # imgL = cv.resize(imgL, (960,540))
        # imgR = cv.resize(imgR, (960,540))
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(imgL,imgleftstr, (50,50), font,1, (0,0,255), 2)
        cv.putText(imgL, imgloopstr, (50,100),font,1, (0,0,255), 2)
        cv.putText(imgR, imgrightstr, (50, 50), font, 1, (0, 0, 255), 2)
        cv.putText(imgR, imgloopstr, (50, 100), font, 1, (0, 0, 255), 2)
        combined = np.concatenate((imgL, imgR), axis=1)
        cv.imshow('img left', combined)
        # cv.waitKey(5000)
        if cv.waitKey(1000) & 0xFF == ord('q'):
            break

cv.destroyAllWindows()

