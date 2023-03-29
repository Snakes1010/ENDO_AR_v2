import cv2 as cv
import os
import glob
import AR_functions

chessboardSize = (15,15)

images_left = glob.glob('/Users/jcsimon/Desktop/ENDO_AR/frames_7_14/frameL/*')
images_right = glob.glob('/Users/jcsimon/Desktop/ENDO_AR/frames_7_14/frameR/*')
images_left_sort = sorted(images_left)
images_right_sort = sorted(images_right)

objpoints_L, imgpoints_L = Checker_Board_functions.calibrate_fast(images_left_sort, chessboardSize)
cv.destroyAllWindows()
objpoints_R, imgpoints_R = Checker_Board_functions.calibrate_fast(images_right_sort, chessboardSize)
cv.destroyAllWindows()