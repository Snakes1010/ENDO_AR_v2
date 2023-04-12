# Use this to calibrate individual cameras and make yaml files for each camera
import cv2 as cv
import glob
import yaml
import os
import numpy as np

import AR_functions
import AR_functions as AR
# from pydub import AudioSegment
# from pydub.playback import play

# sound_search_CB = AudioSegment.from_mp3('/Users/jacobsimon/Desktop/ENDO_AR/soundeffects/ui_prompt_finish_complete_warm_tone.mp3')
# sound_start_calibration = AudioSegment.from_mp3('/Users/jacobsimon/Desktop/ENDO_AR/soundeffects/pop_short_simple.mp3')
# sound_calibrated = AudioSegment.from_mp3('/Users/jacobsimon/Desktop/ENDO_AR/soundeffects/bubbles_popping_ascending_fast.mp3')

chessboardSize = (7,10)

# if not os.path.exists(dir_name):
#     os.makedirs(dir_name)
# grab extracted frames from a folder and add it to a list and sort it
camera_parameters_L_file ='calib_more_angles_L.yaml'
camera_parameters_R_file ='calib_more_angles_R.yaml'


left_chosen_path = AR_functions.import_yaml()
right_chosen_path = AR_functions.import_yaml()


images_left_sort = sorted(left_chosen_path)
images_right_sort = sorted(right_chosen_path)
dimensions = cv.imread(images_left_sort[0])
dimensions = dimensions.shape
#dimensions are (# of rows, # of cols, color depth BGR)
print("FRAME DIMENSIONS:\n", dimensions)
width = dimensions[1]
height = dimensions[0]
print('width:', width)
print('height', height)

objpoints_L, imgpoints_L = AR.calibrate_fine(images_left_sort, chessboardSize)
cv.destroyAllWindows()
objpoints_R, imgpoints_R = AR.calibrate_fine(images_right_sort, chessboardSize)
cv.destroyAllWindows()


################################################################


print('STARTING CAMERA CALIBRATION')

rms_L, camera_matrix_L, dist_L, rvecs_L, tvecs_L = cv.calibrateCamera(objpoints_L, imgpoints_L, (width, height), None, None)
newCameraMatrix_L, roi_L = cv.getOptimalNewCameraMatrix(camera_matrix_L, dist_L, (width,height), 1)
rms_R, camera_matrix_R, dist_R, rvecs_R, tvecs_R = cv.calibrateCamera(objpoints_R, imgpoints_R, (width, height), None, None)
newCameraMatrix_R, roi_R = cv.getOptimalNewCameraMatrix(camera_matrix_R, dist_R, (width, height), 1)
fx_L = newCameraMatrix_L[0, 0]
fy_L = newCameraMatrix_L[1, 1]
fx_R = newCameraMatrix_R[0, 0]
fy_R = newCameraMatrix_R[1, 1]

print("Left camera RMS:", rms_L)
print("Left camera matrix:\n", camera_matrix_L)
print("Left camera distortion coefficients:\n", dist_L)
print("Left camera rotation vectors:\n", rvecs_L)
print("Left camera translation vectors:\n", tvecs_L)

print("Right camera RMS:", rms_R)
print("Right camera matrix:\n", camera_matrix_R)
print("Right camera distortion coefficients:\n", dist_R)
print("Right camera rotation vectors:\n", rvecs_R)
print("Right camera translation vectors:\n", tvecs_R)

print("New left camera matrix:\n", newCameraMatrix_L)
print("ROI for left camera:", roi_L)
print("New right camera matrix:\n", newCameraMatrix_R)
print("ROI for right camera:", roi_R)


print('CALIBRATION SUCCESSFUL')


################################################################

rvecs_L_list = [rvec.tolist() for rvec in rvecs_L]
tvecs_L_list = [tvec.tolist() for tvec in tvecs_L]
rvecs_R_list = [rvec.tolist() for rvec in rvecs_R]
tvecs_R_list = [tvec.tolist() for tvec in tvecs_R]

calib_params_L = {
    'camera_matrix_L': camera_matrix_L.tolist(),
    'new_camera_matrix_L': newCameraMatrix_L.tolist(),
    'dist_coeffs_L': dist_L.tolist(),
    'rvecs_L': rvecs_L_list,
    'tvecs_L': tvecs_L_list,
    'rms_L': float(rms_L),
}
calib_params_R = {
    'camera_matrix_R': camera_matrix_R.tolist(),
    'new_camera_matrix_R': newCameraMatrix_R.tolist(),
    'dist_coeffs_R': dist_R.tolist(),
    'rvecs_R': rvecs_R_list,
    'tvecs_R': tvecs_R_list,
    'rms_R': float(rms_R),
}

with open(camera_parameters_L_file, 'w') as yaml_file:
    yaml.dump(calib_params_L, yaml_file, default_flow_style=False)
with open(camera_parameters_R_file, 'w') as yaml_file:
    yaml.dump(calib_params_R, yaml_file, default_flow_style=False)

print('Calibration Finished!')
