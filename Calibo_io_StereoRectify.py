import cv2 as cv
import glob
import os
import numpy as np
import AR_functions
# identify the location of where we want to save the quality frames

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################


# grab extracted frames from a folder and add it to a list and sort it
images_L = glob.glob('/home/jacob/endo_calib/ENDO_AR/left_calib_one/*')
images_R = glob.glob('/home/jacob/endo_calib/ENDO_AR/right_calib_one/*')
images_sort_L = sorted(images_L)
images_sort_R = sorted(images_R)
chosen = []
chosen_path = []
dimensions = cv.imread(images_sort_L[0])
dimensions = dimensions.shape
#dimensions are (# of rows, # of cols, color depth BGR)
print("FRAME DIMENSIONS:\n", dimensions)
width = 1920
height = 1080
print('width:', width)
print('height', height)

K, k, cam_rvecs, cam_tvecs = AR_functions.readCalibParameters('/home/jacob/endo_calib/camera_calibration_5_26/8_11_5_26/dual_cal.json')

camera_matrix_L = np.array(K[0])
camera_matrix_R = np.array(K[1])
distL = np.array(k[0])
distR = np.array(k[1])
r_vecs = cam_rvecs[1]
t_vecs = cam_tvecs[1]

print('camera_matrix_L:\n', camera_matrix_L)
print('camera_matrix_R:\n', camera_matrix_R)
print('distL:\n', distL)
print('distR:\n', distR)
print('r_vecs:\n', r_vecs)
print('t_vecs:\n', t_vecs)


# StereoVison rectification
rectifyScale = 1
rectL_stereo, rectR_stereo, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(camera_matrix_L, distL,
                                                                                           camera_matrix_R, distR,
                                                                                           (width, height), r_vecs, t_vecs, rectifyScale, (0, 0))
print("rectL_stereo:\n", rectL_stereo)
print("rectR_stereo:\n", rectR_stereo)
print("projMatrixL:\n", projMatrixL)
print("projMatrixR:\n", projMatrixR)
print("Q:\n", Q)
print("roi_L:\n", roi_L)
print("roi_R:\n", roi_R)


stereoMapL = cv.initUndistortRectifyMap(camera_matrix_L, distL, rectL_stereo, projMatrixL,(width, height), cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(camera_matrix_R, distR, rectR_stereo, projMatrixR,(width, height), cv.CV_16SC2)

# print("Saving parameters!")
# cv_file = cv.FileStorage('stereoMap5_26.xml', cv.FILE_STORAGE_WRITE)
#
# cv_file.write('stereoMapL_x', stereoMapL[0])
# cv_file.write('stereoMapL_y', stereoMapL[1])
# cv_file.write('stereoMapR_x', stereoMapR[0])
# cv_file.write('stereoMapR_y', stereoMapR[1])

