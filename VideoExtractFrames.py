import cv2 as cv
import os
import Video_functions

# Input directory to save video frames to
# dir_name_L = "/home/jacob/Desktop/frameL"
# video_L = "/home/jacob/endo_calib/low_cost_proj/8_11_2x/low_res_frame_L.mp4"

dir_name_R = "/home/jacob/Desktop/frameR"
video_R = "//home/jacob/endo_calib/low_cost_proj/8_11_2x/low_res_frame_R.mp4"

# Video_functions.save_video_frames(video_L, dir_name_L)
Video_functions.save_video_frames(video_R, dir_name_R)

cv.destroyAllWindows()