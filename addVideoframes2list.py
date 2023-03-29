# This program takes a video file from a directory and reads the frames in a while lopp
# If the videos are not read correctly it reports and erros
# If both video captures are opened correctly then it appends them to a list

import cv2 as cv
import os
import numpy as np
# identify the location of where we want to save the quality frames
dir_name = "/Users/jcsimon/Desktop/ENDO_AR/data_2_3_23"
# if not os.path.exists(dir_name):
#     os.makedirs(dir_name)
# open the left and right camera paths
left_camera = cv.VideoCapture('/Users/jcsimon/Desktop/ENDO_AR/MANTIS_VIDEOS/11x_8x8_1mm/SHGN7_S001_S001_T003_ISO1.MOV')
right_camera = cv.VideoCapture('/Users/jcsimon/Desktop/ENDO_AR/MANTIS_VIDEOS/11x_8x8_1mm/SHGN7_S001_S001_T003_ISO2.MOV')
# get total number of frames for each images
total_frames_left = int(left_camera.get(cv.CAP_PROP_FRAME_COUNT))
total_frames_right = int(right_camera.get(cv.CAP_PROP_FRAME_COUNT))
print(total_frames_right)
# initialize empty lists for the left and right video frames
frames_left = []
frames_right = []
# initialize a counter to frame = 0-
currentframe = 0
# read throught the left and right videos storing the frames in a lists
while True:
    currentframe += 1
    progress = (currentframe/total_frames_left)*100
    print(str(progress) + "%")
    ret_l, frame_l = left_camera.read()
    ret_r, frame_r = right_camera.read()
    if not ret_l:
        print("ERROR: left camera not read")
        break
    if not ret_r:
            print("ERROR: right camera not read")
            break
    frames_left.append(frame_l)
    frames_right.append(frame_r)

left_camera.release()
right_camera.release()