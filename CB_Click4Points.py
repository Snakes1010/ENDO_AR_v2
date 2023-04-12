# This opens two images from a file and presents them with the checkerboard algo
# the user can click on the corresponding point to return their location

import numpy as np
import cv2 as cv
import glob
import time

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################
chessboardSize = (7,10)
Scalefactor = 1
frameSize = (int(1920/Scalefactor), int(1080/Scalefactor))
print(frameSize)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
xv, yv = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]]
objp[:, :2] = np.array([yv,xv]).T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.

imgL = cv.imread('/Users/jcsimon/Desktop/ENDO_AR/mantis_1mm_8x11/frameL/frame0000.jpg')
imgR = cv.imread('/Users/jcsimon/Desktop/ENDO_AR/mantis_1mm_8x11/frameR/frame0000.jpg')

# function to display the coordinates of
# of the points clicked on the image

def click_eventL(event, x, y, flags, params):

# checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        # displaying the coordinates
        # on the image window
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(imgL, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv.imshow('img left', imgL)


# checking for right mouse clicks
    if event==cv.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        # displaying the coordinates
        # on the image window
        font = cv.FONT_HERSHEY_SIMPLEX
        b = imgL[y, x, 0]
        g = imgL[y, x, 1]
        r = imgL[y, x, 2]
        cv.putText(imgL, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x,y), font, 1,
                    (255, 255, 0), 2)
        cv.imshow('image left', imgL)

def click_eventR(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell

        print(x, ' ', y)
        # displaying the coordinates
        # on the image window
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(imgR, str(x) + ',' +
                   str(y), (x, y), font,
                   1, (255, 0, 0), 2)
        cv.imshow('img right', imgR)
        if event == cv.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)
            # displaying the coordinates
            # on the image window
            font = cv.FONT_HERSHEY_SIMPLEX
            b = imgR[y, x, 0]
            g = imgR[y, x, 1]
            r = imgR[y, x, 2]
            cv.putText(imgR, str(b) + ',' +
                       str(g) + ',' + str(r),
                       (x, y), font, 1,
                       (255, 255, 0), 2)
            cv.imshow('img right', imgR)

imgL = cv.resize(imgL, frameSize)
imgR = cv.resize(imgR, frameSize)
grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
# Find the chess board corners
retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

# If found, add object points, image points (after refining them)
if retL and retR == True:
    # print('matchedpairs\n', imgLeft, imgRight)
    # print(imgloop)
    objpoints.append(objp)
    cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
    imgpointsL.append(cornersL)

    cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
    imgpointsR.append(cornersR)

    # Draw and display the corners
    cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
    cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
    imgL = cv.resize(imgL, frameSize)
    imgR = cv.resize(imgR, frameSize)

    cv.imshow('img left', imgL)
    cv.imshow('img right', imgR)
    cv.setMouseCallback('img left', click_eventL)
    cv.setMouseCallback('img right', click_eventR)
    print('l\n', imgpointsL)
    print('o\n', objpoints)
    cv.waitKey(0)

cv.destroyAllWindows()