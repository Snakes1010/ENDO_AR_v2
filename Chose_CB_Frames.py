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
images_left = glob.glob('/home/jacob/endo_calib/camera_calibration_5_26/charuco/frameL*')
images_right = glob.glob('/home/jacob/endo_calib/camera_calibration_5_26/charuco/frameR*')
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

file_name_L = '5_26_charuco_L.yaml'
file_name_R = '5_26_charuco_R.yaml'

left_chosen_path, right_chosen_path = AR_functions.choose_stereo_pairs(images_left_sort,
                                                                                  images_right_sort,
                                                                                  chessboardSize, resize_factor=.75)

AR_functions.export_yaml(file_name_L, left_chosen_path)
AR_functions.export_yaml(file_name_R, right_chosen_path)


