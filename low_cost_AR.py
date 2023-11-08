import cv2 as cv
import numpy as np
import os

################################################################
# REMAPING
cv_file = cv.FileStorage()
cv_file.open('lowcost.xml', cv.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

################################################################
# AR with google cardboard
AR_use = False
if AR_use == True:
    cv.namedWindow('AR')
    scale_percent = 150  # percent of original size, 200 makes the image twice as large
    width = int((1280) * scale_percent / 100)
    height = int(480 * scale_percent / 100)
    dim = (width, height)
    print(dim)
else:
    cv.namedWindow('AR')
################################################################
# BLOCK MATCHER

stereoMatcher = cv.StereoSGBM_create()
stereoMatcher.setMinDisparity(4)
stereoMatcher.setNumDisparities(16)
stereoMatcher.setBlockSize(5)
# stereoMatcher.setROI1(leftROI)
# stereoMatcher.setROI2(rightROI)
stereoMatcher.setSpeckleRange(44)
stereoMatcher.setSpeckleWindowSize(7)
adjusted_depth = 2000
cv.namedWindow('Depth')

# Define a callback function for the trackbars
def adjust_min_disparity(val):
    print(f"minimum disparity = {val}")
    stereoMatcher.setMinDisparity(val)
def adjust_num_disparities(val):
    val = val * 16  # has to be multiple of 16
    if val == 0:
        val =16
    print(f"number of disparities = {val}")
    stereoMatcher.setNumDisparities(val)
def adjust_block_size(val):
    val = val * 2 + 1  # must be odd and greater than 5
    if val < 5:
        val=5
    print(f"block size = {val}")
    stereoMatcher.setBlockSize(val)
def adjust_speckle_range(val):
    print(f"speck range = {val}")
    stereoMatcher.setSpeckleRange(val)
def adjust_speckle_window_size(val):
    print(f"speckle window = {val}")
    stereoMatcher.setSpeckleWindowSize(val)
def adjusted_depth_range(val):
    adjusted_depth = val
# Create trackbars for 'minDisparity', 'numDisparities' and 'blockSize'.
cv.createTrackbar('Min Disparity', 'Depth', 4, 50, adjust_min_disparity)
cv.createTrackbar('Num Disparities', 'Depth', 1, 16, adjust_num_disparities)  # the max value is set to 16*16
cv.createTrackbar('Block Size', 'Depth', 5, 50, adjust_block_size)  # the max value is set to 50*2+1
cv.createTrackbar('Speckle Range', 'Depth', 44, 50, adjust_speckle_range)
cv.createTrackbar('Speckle Window Size', 'Depth', 7, 200, adjust_speckle_window_size)
cv.createTrackbar('adjusted depth', 'Depth', 2000, 10000, adjusted_depth_range)


#################################################################
#CAMERA SET UP

left_cam = cv.VideoCapture(0)
right_cam = cv.VideoCapture(2)

print_shape_once = True

properties = [cv.CAP_PROP_POS_MSEC, cv.CAP_PROP_POS_FRAMES, cv.CAP_PROP_POS_AVI_RATIO,
              cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT, cv.CAP_PROP_FPS,
              cv.CAP_PROP_FOURCC, cv.CAP_PROP_FRAME_COUNT, cv.CAP_PROP_FORMAT,
              cv.CAP_PROP_MODE, cv.CAP_PROP_BRIGHTNESS, cv.CAP_PROP_CONTRAST,
              cv.CAP_PROP_SATURATION, cv.CAP_PROP_HUE, cv.CAP_PROP_GAIN,
              cv.CAP_PROP_EXPOSURE]

# Property names
property_names = ["POS_MSEC", "POS_FRAMES", "POS_AVI_RATIO", "FRAME_WIDTH",
                  "FRAME_HEIGHT", "FPS", "FOURCC", "FRAME_COUNT", "FORMAT", "MODE",
                  "BRIGHTNESS", "CONTRAST", "SATURATION", "HUE", "GAIN", "EXPOSURE"]

frame_size = (640,480)

for prop in properties:
    valueL = left_cam.get(prop)
    valueR = right_cam.get(prop)
    successL = left_cam.set(prop, valueL)
    successR = right_cam.set(prop, valueR)
    if successL and successR:
        print(f"Property Left {prop} set to {valueL}")
        print(f"Property Right {prop} set to {valueR}")
    else:
        print(f"Failed to set property {prop}")

########################################################################
# RUNS CAMERAS
while True:
    ret_L, frame_L = left_cam.read()
    ret_R, frame_R = right_cam.read()
    if not ret_L or not ret_R:
        print('ERROR: Could not open stereo rig')
        break

    frame_left_remap = cv.remap(frame_L, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT,0)
    frame_right_remap = cv.remap(frame_R, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4,cv.BORDER_CONSTANT, 0)
    gray_left_remap = cv.cvtColor(frame_left_remap, cv.COLOR_BGR2GRAY)
    gray_right_remap = cv.cvtColor(frame_right_remap, cv.COLOR_BGR2GRAY)
    depth = stereoMatcher.compute(gray_left_remap,gray_right_remap)


    if AR_use == True:
        combined_rectify = np.concatenate((frame_left_remap, frame_right_remap), axis=1)
        resized = cv.resize(combined_rectify, dim, interpolation=cv.INTER_AREA)

        cv.line(resized, (0,(360)), (1920, 360),(255,0,255), 3)
        print(resized.shape)
        cv.imshow('AR', resized)
        cv.moveWindow('AR', 5600, 150)
        cv.imshow('Depth', depth/2000)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        combined_rectify = np.concatenate((frame_left_remap, frame_right_remap), axis=1)
        cv.imshow('AR', combined_rectify)
        cv.imshow('Depth', depth / 2000)
        if cv.waitKey(1) & 0xFF == ord('q'):
                break

left_cam.release()
right_cam.release()
cv.destroyAllWindows()
