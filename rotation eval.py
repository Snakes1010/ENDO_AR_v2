import cv2 as cv
import glob
import os
import numpy as np
import AR_functions



grid = cv.imread('/home/jacob/endo_calib/camera_calibration_5_26/8_11_5_26/framesR/frame0001.jpg')
dimensions = grid.shape
#dimensions are (# of rows, # of cols, color depth BGR)
print("FRAME DIMENSIONS:\n", dimensions)
width = dimensions[1]
height = dimensions[0]
print('width:', width)
print('height', height)

K, k, cam_rvecs, cam_tvecs = AR_functions.readCalibParameters('/home/jacob/endo_calib/low_cost_proj/8_11_2x/low_cost_dual.json')

camera_matrix_L = np.array(K[0])
camera_matrix_R = np.array(K[1])
distL = np.array(k[0][:4])
distR = np.array(k[1][:4])
r_vecs = cam_rvecs[1]
t_vecs = cam_tvecs[1]

rx = r_vecs[0]
ry = r_vecs[1]
rz = r_vecs[2]


print('camera_matrix_L:\n', camera_matrix_L)
print('camera_matrix_R:\n', camera_matrix_R)
print('distL:\n', distL)
print('distR:\n', distR)
print('r_vecs:\n', r_vecs)
print('t_vecs:\n', t_vecs)

theta = np.sqrt(rx**2 + ry**2 + rz**2)
axis = np.array([rx, ry, rz]) / theta
rotation_matrix, _ = cv.Rodrigues(theta * axis)
M = np.zeros((2, 3))
M[:2, :2] = rotation_matrix[:2, :2]
# rotation_matrix_transpose = rotation_matrix.T
# print('rotation_matrix transpose:\n', rotation_matrix_transpose)
print('3x3 rotation matrix:\n', rotation_matrix)

rotated = cv.warpAffine(grid, M, (width,height))

cv.imshow('og', grid)
cv.imshow('rot', rotated)

cv.waitKey(0)
cv.destroyAllWindows()