import cv2 as cv
import glob
import os
import numpy as np
import AR_functions
# identify the location of where we want to save the quality frames
dir_name = ""
camera_parameters_L_file ='calib_100_L3.yaml'
camera_parameters_R_file ='calib_100_R3.yaml'

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################
# chessboardSize = (row, col)
chessboardSize = (10,7)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (0,1,0), (0,2,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
xv, yv = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]]
objp[:, :2] = np.array([yv,xv]).T.reshape(-1,2)
scalefactor = 1

# Arrays to store object points and image points from all the images.
objpoints_L = [] # 3d point in real world space
objpoints_R = [] # 3d point in real world space
imgpoints_L = [] # 2d points in image plane.
imgpoints_R = [] # 2d points in image plane.
camera_matrix_L = []
camera_matrix_R = []
distL = []
distR = []

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
width = dimensions[1]
height = dimensions[0]
print('width:', width)
print('height', height)

K, k, cam_rvecs, cam_tvecs = AR_functions.readCalibParameters('/home/jacob/endo_calib/ENDO_AR/cam_calib_nodist.json')

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



# left_chosen_path, right_chosen_path = AR_functions.choose_stereo_pairs(images_sort_L, images_sort_R, chessboardSize)


# objpoints_L, imgpoints_L = AR_functions.calibrate_fine(images_sort_L, chessboardSize)
# print('left calibrated')
# objpoints_R, imgpoints_R = AR_functions.calibrate_fine(images_sort_R, chessboardSize)
# print('right calibrated')
#
# flags = 0
# flags |= cv.CALIB_FIX_INTRINSIC
# # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# # Hence intrinsic parameters are the same
# criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# # This step is performed to transformation between the two cameras and calculate Essential and Fundamental matrix
# # Size of the image used only to initialize the intrinsic camera matrix [w,h].
# retStereo, stereoCameraMatrixL, distL_stereo, stereoCameraMatrixR, distR_stereo, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints_L, imgpoints_L, imgpoints_R,
# camera_matrix_L, distL, camera_matrix_R, distR, (width, height), criteria_stereo, flags)
#
# print('retStereo:', retStereo)
# print('stereoCameraMatrixL:', stereoCameraMatrixL)
# print('distL_stereo:', distL_stereo)
# print('stereoCameraMatrixR:', stereoCameraMatrixR)
# print('distR_stereo:', distR_stereo)
# print('rot:', rot)
# print('trans:', trans)
# print('essentialMatrix:', essentialMatrix)
# print('fundamentalMatrix:', fundamentalMatrix)


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

print("Saving parameters!")
cv_file = cv.FileStorage('stereoMap5_12.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x', stereoMapL[0])
cv_file.write('stereoMapL_y', stereoMapL[1])
cv_file.write('stereoMapR_x', stereoMapR[0])
cv_file.write('stereoMapR_y', stereoMapR[1])

