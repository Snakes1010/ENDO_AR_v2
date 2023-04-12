import cv2 as cv
import os
import Video_functions

# Input directory to save video frames to
dir_name_L = "/Users/jcsimon/Desktop/ENDO_AR/mantis_1mm_8x11/frameL"
video_L = '/Users/jcsimon/Desktop/ENDO_AR/mantis_1mm_8x11/SHGN7_S001_S001_T003_ISO1.MOV'

dir_name_R = "/Users/jcsimon/Desktop/ENDO_AR/mantis_1mm_8x11/frameR"
video_R = '/Users/jcsimon/Desktop/ENDO_AR/mantis_1mm_8x11/SHGN7_S001_S001_T003_ISO2.MOV'

Video_functions.save_video_frames(video_L, dir_name_L)
Video_functions.save_video_frames(video_R, dir_name_R)

cv.destroyAllWindows()