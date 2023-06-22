import cv2 as cv
import glob
import numpy as np
import AR_functions

chessboardSize = (10,7)
# cv_file = cv.FileStorage()
# cv_file.open('stereoMap6_6.xml', cv.FileStorage_READ)
#
# stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
# stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
# stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
# stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

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
images_L = glob.glob('/home/jacob/endo_calib/camera_calibration_5_26/8_11_5_26/chosen_framesL2/*')
images_R = glob.glob('/home/jacob/endo_calib/camera_calibration_5_26/8_11_5_26/chosen_framesR2/*')
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

K, k, cam_rvecs, cam_tvecs = AR_functions.readCalibParameters('/home/jacob/endo_calib/camera_calibration_5_26/8_11_5_26/5param_dual.json')

camera_matrix_L = np.array(K[0])
camera_matrix_R = np.array(K[1])
distL = np.array(k[0][:5])
distR = np.array(k[1][:5])
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



original_L = cv.imread(images_sort_L[35])
original_R = cv.imread(images_sort_R[35])

print('3x3 rotation matrix:\n', rotation_matrix)

undistorted_image_L = cv.undistort(original_L, camera_matrix_L, distL)
undistorted_image_R = cv.undistort(original_R, camera_matrix_R, distR)



gray_img_L = cv.cvtColor(undistorted_image_L, cv.COLOR_BGR2GRAY)
gray_img_R = cv.cvtColor(undistorted_image_R, cv.COLOR_BGR2GRAY)
ret, cornersL = cv.findChessboardCornersSB(gray_img_L, chessboardSize, flags=None)
ret, cornersR = cv.findChessboardCornersSB(gray_img_R, chessboardSize, flags=None)


fundamental_matrix, inl = cv.findFundamentalMat(cornersL, cornersR, cv.FM_RANSAC)
pointL = cornersL[:,0:1]
point1 = pointL[0:3]
pointR = cornersR[:,0:1]
point2 = pointR[0:3]
print(fundamental_matrix)
H1, H2 = cv.stereoRectifyUncalibrated(point1, point2, fundamental_matrix, (1920,1080))

print(H1)
print(H2)
# undistorted_image_L = cv.remap(undistorted_image_L, stereoMapL_x, stereoMapL_y, interpolation=cv.INTER_LINEAR)
# undistorted_image_R = cv.remap(undistorted_image_R, stereoMapR_x, stereoMapR_y, interpolation=cv.INTER_LINEAR)

# og = np.concatenate((original_L,original_R) ,axis=1)
undist = np.concatenate((undistorted_image_L,undistorted_image_R), axis=1)


cv.imshow("original image", og)
cv.imshow('undistorted image', undist)
cv.waitKey(0)
cv.destoryAllWindows()