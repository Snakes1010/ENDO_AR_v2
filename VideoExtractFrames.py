import cv2 as cv
import os
import Video_functions

# Input directory to save video frames to
dir_name_L = "/home/jacob/endo_calib/camera_calibration_5_26/charuco/frameL"
video_L = "/home/jacob/endo_calib/camera_calibration_5_26/charuco/SHGN7_S001_S001_T012/SHGN7_S001_S001_T012_ISO1.MOV"

dir_name_R = "/home/jacob/endo_calib/camera_calibration_5_26/charuco/frameR"
video_R = "/home/jacob/endo_calib/camera_calibration_5_26/charuco/SHGN7_S001_S001_T012/SHGN7_S001_S001_T012_ISO2.MOV"

Video_functions.save_video_frames(video_L, dir_name_L)
Video_functions.save_video_frames(video_R, dir_name_R)

cv.destroyAllWindows()