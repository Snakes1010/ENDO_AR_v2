import cv2 as cv
import glob
import os
import numpy as np

# identify the location of where we want to save the quality frames
dir_name = ""
camera_parameters_L_file ='calib_100_L3.yaml'
camera_parameters_R_file ='calib_100_R3.yaml'

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################
# chessboardSize = (row, col)
chessboardSize = (7,10)
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

# grab extracted frames from a folder and add it to a list and sort it
images_L = glob.glob('/Users/jcsimon/Desktop/ENDO_AR/data_2_3_23/frameL/*')
images_R = glob.glob('/Users/jcsimon/Desktop/ENDO_AR/data_2_3_23/frameR/*')
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

if os.path.exists(camera_parameters_L_file) or os.path.exists(camera_parameters_R_file):
    with open(camera_parameters_L_file, 'r') as file:
        params_dict_L = yaml.load(file, Loader=yaml.FullLoader)

    camera_matrix_L = np.array(params_dict_L['camera_matrix_L'])
    dist_coeffs_L = np.array(params_dict_L['dist_coeffs_L'])
    rms_L = params_dict_L['rms_L']
    rvecs_L = np.array(params_dict_L['rvecs_L'])
    tvecs_L = np.array(params_dict_L['tvecs_L'])

    with open(camera_parameters_R_file, 'r') as file:
        params_dict_R = yaml.load(file, Loader=yaml.FullLoader)

    camera_matrix_R = np.array(params_dict_R['camera_matrix_R'])
    dist_coeffs_R = np.array(params_dict_R['dist_coeffs_R'])
    rms_R = np.array(params_dict_R['rms_R'])
    rvecs_R = np.array(params_dict_R['rvecs_R'])
    tvecs_R = np.array(params_dict_R['tvecs_R'])
else:
    print("No camera parameters found")

imgloop = 0
for img_L in images_sort_L:
    imgloop += 1
    img_L = cv.imread(img_L)
    gray_screen_L = cv.cvtColor(img_L, cv.COLOR_BGR2GRAY)
    ret_L, corners_L = cv.findChessboardCorners(gray_screen_L, chessboardSize, None, flags=cv.CALIB_CB_FAST_CHECK)
    if ret_L == True:
        objpoints_L.append(objp)
        imgpoints_L.append(corners_L)
        cv.drawChessboardCorners(img_L, chessboardSize, corners_L, ret_L)
        cv.putText(img_L, str(imgloop), (50, 100), font, 1, (0, 0, 255), 2)
        cv.imshow('img', img_L)
        key = cv.waitKey(1) & 0xFF
        chosen.append(img_L)
        if key == ord('q'):
            break
imgloop = 0
for img_R in images_sort_R:
    imgloop += 1
    img1 = cv.imread(img)
    gray_screen_L = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    ret_L, corners_L = cv.findChessboardCorners(gray_screen_L, chessboardSize, None, flags=cv.CALIB_CB_FAST_CHECK)
    if ret_L == True:
        objpoints_L.append(objp)
        imgpoints_L.append(corners_L)
        cv.drawChessboardCorners(img1, chessboardSize, corners_L, ret_L)
        cv.putText(img1, str(imgloop), (50, 100), font, 1, (0, 0, 255), 2)
        cv.imshow('img', img1)
        key = cv.waitKey(1) & 0xFF
        chosen.append(img)
        if key == ord('q'):
            break
print('done')
cv.destroyAllWindows()
cv.waitKey(1)

print('imgpointL\n', imgpointsL)
print('objpoints\n', objpoints)

############# CALIBRATION #######################################################
imageSize Size of the image used only to initialize the intrinsic camera matrix [w,h].
ret, cameraMatrix_L, dist_L, rvecs_L, tvecs_L = cv.calibrateCamera(objpoint_s, imgpoints_L, (width, height), None, None)
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (width,height), 1, (width,height))

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (width,height), 1, (width,height))

print('newCameraMatrixL:\n', newCameraMatrix)
print('rmse_l:', ret)

if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

out_filename = os.path.join('camera_parametersL' + '3_8_23_'+'intrinsics.dat')
outf = open(out_filename, 'w')

outf.write('intrinsic:\n')
for l in newCameraMatrix:
    for en in l:
        outf.write(str(en) + ' ')
    outf.write('\n')

outf.write('distortion:\n')
for en in dist[0]:
    outf.write(str(en) + ' ')
outf.write('\n')

outf.write('rotation:\n')
for en in rvecs[0]:
    outf.write(str(en) + ' ')
outf.write('\n')

outf.write('translation:\n')
for en in tvecs[0]:
    outf.write(str(en) + ' ')
outf.write('\n')

# StereoVison rectification
rectifyScale = 1
rectL_stereo, rectR_stereo, projMatrixL, projMatrixR, Q, roi_L2, roi_R2 = cv.stereoRectify(StereoCameraMatrixL, distL_stereo,
                                                                                           StereoCameraMatrixR, distR_stereo,
                                                                                           (height, width), rot, trans, rectifyScale, (0, 0))

stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL_stereo, rectL_stereo, projMatrixL,(height, width), cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR_stereo, rectR_stereo, projMatrixR,(height, width), cv.CV_16SC2)

print("Saving parameters!")
cv_file = cv.FileStorage('stereoMap2_24_23.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x', stereoMapL[0])
cv_file.write('stereoMapL_y', stereoMapL[1])
cv_file.write('stereoMapR_x', stereoMapR[0])
cv_file.write('stereoMapR_y', stereoMapR[1])

print(grayR_chosen.shape)
print(grayR_chosen.shape[::-1])