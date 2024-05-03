import cv2 as cv
import glob
import os
import numpy as np
import AR_functions

print('FRAME DIMENSIONS:')
width = 640
height = 480
print('width:', width)
print('height', height)

K, k, cam_rvecs, cam_tvecs = AR_functions.readCalibParameters('/media/jacob/Viper4TB/minicam_calibration/dual_cam_calib.json')

camera_matrix_L = np.array(K[0])
camera_matrix_R = np.array(K[1])
distL = np.array(k[0][:4])
distR = np.array(k[1][:4])
r_vecs = cam_rvecs[1]
t_vecs = cam_tvecs[1]

rx = r_vecs[0]
ry = r_vecs[1]
rz = r_vecs[2]

theta = np.sqrt(rx**2 + ry**2 + rz**2)
# Compute the axis of rotation
if theta != 0:
    axis = np.array([rx, ry, rz]) / theta
else:
    axis = np.array([0, 0, 0])  # Set default axis if theta is zero
# Convert axis-angle representation to rotation matrix
rotation_matrix, _ = cv.Rodrigues(theta * axis)


print('camera_matrix_L:\n', camera_matrix_L)
print('camera_matrix_R:\n', camera_matrix_R)
# print('distL:\n', distL)
# print('distR:\n', distR)
# print('r_vecs:\n', r_vecs)
# print('rotation_matrix:\n', rotation_matrix)
# print('t_vecs:\n', t_vecs)

flags = 0
R1, R2, P1, P2, Q, roi_L, roi_R = cv.stereoRectify(camera_matrix_L, distL, camera_matrix_R, distR,
                                                                     (width, height), rotation_matrix, t_vecs, flags=flags)
print("R1:\n", R1)
print("R2:\n", R2)
print("projMatrixL:\n", P1)
print("projMatrixR:\n", P2)
print("Q:\n", Q)
print("roi_L:\n", roi_L)
print("roi_R:\n", roi_R)

fs = cv.FileStorage('Q_dual_5_2.yaml', cv.FILE_STORAGE_WRITE)
fs.write('Q', Q)
fs.release()

stereoMapL = cv.initUndistortRectifyMap(camera_matrix_L, distL, R1, P1,(width, height), cv.CV_32FC1)
stereoMapR = cv.initUndistortRectifyMap(camera_matrix_R, distR, R2, P2,(width, height), cv.CV_32FC1)


print("Saving parameters!")
cv_file = cv.FileStorage('dualcam_5_2.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x', stereoMapL[0])
cv_file.write('stereoMapL_y', stereoMapL[1])
cv_file.write('stereoMapR_x', stereoMapR[0])
cv_file.write('stereoMapR_y', stereoMapR[1])
print("Parameters saved!")

camera_matrix_L, rotMatrixL, transVectorL, rotMatrixX_L, rotMatrixY_L, rotMatrixZ_L, euler_angles_L = cv.decomposeProjectionMatrix(P1)
camera_matrix_R, rotMatrixR, transVectorR, rotMatrixX_R, rotMatrixY_R, rotMatrixZ_R, euler_angles_R = cv.decomposeProjectionMatrix(P2)


# Extract rectified intrinsic parameters
focal_length_x = P1[0, 0]
focal_length_y = P1[1, 1]
principal_point_x = P1[0, 2]
principal_point_y = P1[1, 2]

# Display results
print("Camera Matrix Left:")
print(camera_matrix_L)
print("Camera Matrix Right:")
print(camera_matrix_R)
print("Rectification Matrix Left:")
print(R1)
print("Rectification Matrix Right:")
print(R2)
print("Rectified Intrinsic Matrix Left:")
print(P1)
print("Rectified Intrinsic Matrix Right:")
print(P2)
print('focal_length_x')
print(focal_length_x)
print('focal_length_y')
print(focal_length_y)
print('principal_point_x')
print(principal_point_x)
print('principal_point_y')
print(principal_point_y)