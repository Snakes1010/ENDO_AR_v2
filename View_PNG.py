import cv2 as cv
import numpy as np

image = cv.imread('/media/jacob/SNAKE_2TB/Blender_bin/Endo_scans/stereo_test_images/color_000001_L.png')
cv.imshow('iamge',image)
cv.waitKey(10000)