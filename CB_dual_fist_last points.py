import cv2 as cv
import glob
import yaml
import os
import numpy as np
import AR_functions
import random

chessboardSize = (10,7)

# if not os.path.exists(dir_name):
#     os.makedirs(dir_name)q
# grab extracted frames from a folder and add it to a list and sort it
camera_parameters_L_file ='calib_200_L.yaml'
camera_parameters_R_file ='calib_200_R.yaml'
stereo_parameters_file ='stereo_calib_params.yml'
stereo_map_file = ''
images_left = glob.glob('/Users/jacobsimon/Desktop/ENDO_AR/data_2_3_23/frameL/*')
images_right = glob.glob('/Users/jacobsimon/Desktop/ENDO_AR/data_2_3_23/frameR/*')
images_left_sort = sorted(images_left)
images_right_sort = sorted(images_right)
dimensions = cv.imread(images_left_sort[0])
dimensions = dimensions.shape
#dimensions are (# of rows, # of cols, color depth BGR)
print("FRAME DIMENSIONS:\n", dimensions)
width = dimensions[1]
height = dimensions[0]
print('width:', width)
print('height', height)
img_size_w_h = (width, height)

if os.path.exists(camera_parameters_L_file) or os.path.exists(camera_parameters_R_file):
    with open(camera_parameters_L_file, 'r') as file:
        params_dict_L = yaml.load(file, Loader=yaml.FullLoader)

    camera_matrix_L = np.array(params_dict_L['camera_matrix_L'])
    new_camera_matrix_L = np.array(params_dict_L['new_camera_matrix_L'])
    dist_coeffs_L = np.array(params_dict_L['dist_coeffs_L'])
    rms_L = params_dict_L['rms_L']
    rvecs_L = np.array(params_dict_L['rvecs_L'])
    tvecs_L = np.array(params_dict_L['tvecs_L'])

    with open(camera_parameters_R_file, 'r') as file:
        params_dict_R = yaml.load(file, Loader=yaml.FullLoader)

    camera_matrix_R = np.array(params_dict_R['camera_matrix_R'])
    new_camera_matrix_R = np.array(params_dict_R['new_camera_matrix_R'])
    dist_coeffs_R = np.array(params_dict_R['dist_coeffs_R'])
    rms_R = np.array(params_dict_R['rms_R'])
    rvecs_R = np.array(params_dict_R['rvecs_R'])
    tvecs_R = np.array(params_dict_R['tvecs_R'])
else:
    print("No camera parameters found")
print('INDIVIDUAL CAMERA CALIBRATION ----- Done!')

if os.path.exists(stereo_parameters_file):
    with open(stereo_parameters_file, 'r') as file:
        stereo_dict = yaml.load(file, Loader=yaml.FullLoader)

    retStereo = stereo_dict['retStereo']
    StereoCameraMatrixL = np.array(stereo_dict['StereoCameraMatrixL'])
    distL_stereo = np.array(stereo_dict['distR_stereo'])
    StereoCameraMatrixR = np.array(stereo_dict['StereoCameraMatrixR'])
    distR_stereo = np.array(stereo_dict['distR_stereo'])
    rot = np.array(stereo_dict['rot'])
    trans = np.array(stereo_dict['trans'])
    essentialMatirx = np.array(stereo_dict['essentialMatrix'])
    fundamentalMatrix = np.array(stereo_dict['fundamentalMatrix'])

else:
    print("No stereo parameters found")

    left_chosen_path, right_chosen_path = Checker_Board_functions.choose_stereo_pairs(images_left_sort,
                                                                                      images_right_sort,
                                                                                      chessboardSize)

    objpoints_L, imgpoints_L = Checker_Board_functions.calibrate_fine(left_chosen_path, chessboardSize)
    cv.destroyAllWindows()
    objpoints_R, imgpoints_R = Checker_Board_functions.calibrate_fine(right_chosen_path, chessboardSize)
    cv.destroyAllWindows()

    retStereo, StereoCameraMatrixL, distL_stereo, StereoCameraMatrixR, \
        distR_stereo, rot, trans, essentialMatrix, fundamentalMatrix = \
        Checker_Board_functions.stereo_calibrate(objpoints_L, imgpoints_L,imgpoints_R,
                                             new_camera_matrix_L, dist_coeffs_L, new_camera_matrix_R, dist_coeffs_L,
                                             width, height)
print('STEREO_CALIBRATION --- Done!')

if os.path.exists(stereo_map_file):
    img_left = cv.imread(images_left_sort[0])
    img_right = cv.imread(images_right_sort[0])
    cv.imshow('left_og', img_left)
    cv.imshow('right_og', img_right)
    # rectify the left and right images using the rectification maps
    rectified_left = cv.remap(img_left, stereoMapL_x, stereoMapL_y, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
    rectified_right = cv.remap(img_right, stereoMapR_x, stereoMapR_y, cv.INTER_LINEAR, cv.BORDER_CONSTANT)

    # display the rectified images
    cv.imshow('Rectified Left', rectified_left)
    cv.imshow('Rectified Right', rectified_right)
    cv.waitKey(0)
    cv.destroyAllWindows()


else:
    print("No stereo map found")
    stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y = \
        Checker_Board_functions.rectify_undistort(StereoCameraMatrixL,distL_stereo,
                                                  StereoCameraMatrixR, distR_stereo,img_size_w_h, rot, trans)

    img_left = cv.imread(images_left_sort[0])
    img_right = cv.imread(images_right_sort[0])
    cv.imshow('left_og', img_left)
    cv.imshow('right_og', img_right)
    # rectify the left and right images using the rectification maps
    rectified_left = cv.remap(img_left, stereoMapL_x, stereoMapL_y, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
    rectified_right = cv.remap(img_right, stereoMapR_x, stereoMapR_y, cv.INTER_LINEAR, cv.BORDER_CONSTANT)

    # display the rectified images
    cv.imshow('Rectified Left', rectified_left)
    cv.imshow('Rectified Right', rectified_right)
    cv.waitKey(0)
    cv.destroyAllWindows()