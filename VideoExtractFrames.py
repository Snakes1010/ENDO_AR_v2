import cv2 as cv
import os
import Video_functions

# Input directory to save video frames to
dir_name_L = "/home/jacob/endo_calib/ENDO_AR/mantis_8x11/test"
video_L = '/home/jacob/endo_calib/ENDO_AR/mantis_8x11/SHGN7_S001_S001_T002_ISO1.MOV'

dir_name_R = "//home/jacob/endo_calib/ENDO_AR/mantis_8x11/test"
video_R = '/home/jacob/endo_calib/ENDO_AR/mantis_8x11/SHGN7_S001_S001_T002_ISO2.MOV'

Video_functions.save_video_frames(video_L, dir_name_L)
Video_functions.save_video_frames(video_R, dir_name_R)

cv.destroyAllWindows()