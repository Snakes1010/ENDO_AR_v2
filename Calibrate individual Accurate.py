import cv2 as cv
import glob
import os
import numpy as np

# identify the location of where we want to save the quality frames
dir_name = ""
font = cv.FONT_HERSHEY_SIMPLEX
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
imgloop = 0
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# grab extracted frames from a folder and add it to a list and sort it
images = glob.glob('/Users/jcsimon/Desktop/ENDO_AR/data_2_3_23/frameR/*')
images_sort = sorted(images)
chosen = []
chosen_path = []
dimensions = cv.imread(images_sort[0])
dimensions = dimensions.shape
#dimensions are (# of rows, # of cols, color depth BGR)
print("FRAME DIMENSIONS:\n", dimensions)
width = dimensions[1]
height = dimensions[0]
print('width:', width)
print('height', height)

for img in images_sort:
    imgloop += 1
    img1 = cv.imread(img)
    gray_screen = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray_screen, chessboardSize, None, flags=cv.CALIB_CB_FAST_CHECK)
    if ret == True:
        cv.drawChessboardCorners(img1, chessboardSize, corners, ret)
        cv.putText(img1, str(imgloop), (50, 100), font, 1, (0, 0, 255), 2)
        cv.imshow('img', img1)
        key = cv.waitKey(1) & 0xFF
        chosen.append(img)
        if key == ord('q'):
            break
print('done')
cv.destroyAllWindows()
cv.waitKey(1)
imgloop = 0
if len(chosen) != 0:
    for img_chosen in chosen:
        imgloop += 1
        img_accurate = cv.imread(img_chosen)
        gray_accurate = cv.cvtColor(img_accurate, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret_accurate, corners_accurate = cv.findChessboardCorners(gray_accurate, chessboardSize, None)
        # If found, add object points, image points (after refining them)
        if ret_accurate  == True:
            objpoints.append(objp)
            # R and L subpixel
            corners_accurate = cv.cornerSubPix(gray_accurate, corners_accurate, (11,11), (-1,-1), criteria)
            imgpoints.append(corners_accurate)
            # Draw and display the corners
            cv.drawChessboardCorners(img_accurate, chessboardSize, corners_accurate, ret_accurate)
            cv.putText(img_accurate, str(imgloop), (50,100),font,1, (0,0,255), 2)
            cv.imshow('Accurate', img_accurate)
            cv.waitKey(1)

    cv.destroyAllWindows()

# print('imgpointL\n', imgpointsL)
# print('objpoints\n', objpoints)

############## CALIBRATION #######################################################
# imageSize Size of the image used only to initialize the intrinsic camera matrix [w,h].
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (width,height), 1, (width,height))

print('newCameraMatrixL:\n', newCameraMatrix)
print('rmse_l:', ret)
print('[2.85265117e+04 0.00000000e+00 9.58848782e+02],'
      '[0.00000000e+00 2.91584395e+04 5.39000661e+02] '
      '[0.00000000e+00 0.00000000e+00 1.00000000e+00]')

if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

out_filename = os.path.join('camera_parametersR_accurate' + '_intrinsics.dat')
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
