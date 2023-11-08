import sys
import cv2 as cv
import numpy as np
import time
from matplotlib import pyplot as plt

cv_file = cv.FileStorage()
cv_file.open('lowcost.xml', cv.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()


# Open both cameras
# cap_left = cv.VideoCapture('SHGN7_S001_S001_T012_ISO1.MOV')
# cap_right = cv.VideoCapture('SHGN7_S001_S001_T012_ISO2.MOV')
cap_left = cv.VideoCapture('/home/jacob/endo_calib/low_cost_proj/8_11_2x/stereo_video_L.mp4')
cap_right = cv.VideoCapture('/home/jacob/endo_calib/low_cost_proj/8_11_2x/stereo_video_R.mp4')



while(cap_left.isOpened() and cap_right.isOpened()):
        # load color raw image(1080 rows, 1920 columns)
        succes_left, frame_left_col = cap_left.read()
        succes_right, frame_right_col = cap_right.read()

        # convert color to gray scale
        frame_left_gray = cv.cvtColor(frame_left_col, cv.COLOR_BGR2GRAY)
        frame_right_gray = cv.cvtColor(frame_right_col, cv.COLOR_BGR2GRAY)
        # # remap function on gray images
        frame_left_gray_remap = cv.remap(frame_left_gray, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
        frame_right_gray_remap = cv.remap(frame_right_gray, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

        combined_rectify = np.concatenate((frame_left_gray_remap,frame_right_gray_remap), axis=1)
        combined_original = np.concatenate((frame_left_col, frame_right_col), axis=1)
        for i in range(0, combined_rectify.shape[0], 30):
                cv.line(combined_rectify, (0, i), (combined_rectify.shape[1], i), (255, 0, 255), 1)
                cv.line(combined_original, (0, i), (combined_original.shape[1], i), (0, 255, 255), 1)
        cv.imshow('rectified',combined_rectify)
        cv.imshow('original',combined_original)


        # cv.imshow('remap', frame_right_gray_remap)

        key =  cv.waitKey(30) & 0xFF
        if key == ord('q'):
                break
        # stereo = cv.StereoBM_create(numDisparities=32, blockSize=21)
        # depth = stereo.compute(frame_left_gray, frame_right_gray)
        #
        # plt.imshow(depth)
        # plt.axis('off')
        # plt.show()
        # plt.clear()
        ################## CALIBRATION #########################################################
        #
        # frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)

        ########################################################################################

        if cv.waitKey(1) & 0xFF == ord('q'):
                break
# def undistortRectify(frameR, frameL):
#
#     # Undistort and rectify images
#     undistortedL= cv.remap(frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
#     undistortedR= cv.remap(frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
#
#
#     return undistortedR, undistortedL

# Function for stereo vision and depth estimation
# import triangulation as tri
# import calibration_function



# Stereo vision setup parameters
# frame_rate = 60  # Camera frame rate (maximum at 120 fps)
# B = 6.5  # Distance between the cameras [cm]
# f = 22.2  # Camera lense's focal length [mm]
# alpha = 78.07  # Camera field of view in the horisontal plane [degrees]


# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv.destroyAllWindows()