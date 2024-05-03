import cv2 as cv
import numpy as np
import open3d as o3d
from PIL import Image

# REMAPING
cv_file = cv.FileStorage()
cv_file.open('lowcost_2_8.xml', cv.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()


left_endo = Image.open("/home/jacob/Snake2TB/Blender_bin/Endo_scans/blender_batch_feb_3_06_samples/013V9CKV_upper/left/16_1_.png")
right_endo = Image.open("/home/jacob/Snake2TB/Blender_bin/Endo_scans/blender_batch_feb_3_06_samples/013V9CKV_upper/right/16_1_.png")
left_endo = np.array(left_endo)
right_endo = np.array(right_endo)

frame_left_remap = cv.remap(left_endo, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT,0)
frame_right_remap = cv.remap(right_endo, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4,cv.BORDER_CONSTANT, 0)
gray_left_remap = cv.cvtColor(frame_left_remap, cv.COLOR_BGR2GRAY)
gray_right_remap = cv.cvtColor(frame_right_remap, cv.COLOR_BGR2GRAY)

cv.imshow('left', gray_left_remap)
cv.imshow('right', gray_right_remap)
cv.imshow('left endo:', left_endo)
cv.imshow('right endo:', right_endo)

cv.waitKey(5000)
cv.destroyAllWindows()