import cv2 as cv
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import numpy as np
import open3d as o3d
import json

fs = cv.FileStorage('Q_7_25.yaml', cv.FILE_STORAGE_READ)
Q = fs.getNode('Q').mat()
print("Loaded Q:\n", Q)


# Release the FileStorage object
fs.release()

# Import camera calibration data
with open('/media/jacob/SNAKE_2TB/endo_calib/low_cost_proj/8_11_2x/low_cost_dual_Charuco.json', 'r') as file:
    calib = json.load(file)

sensor_width_mm = 8.8
sensor_height_mm = 6.6
sensor_width_px = 640
sensor_height_px = 480

left_cam_model = calib["Calibration"]["cameras"][0]["model"]["ptr_wrapper"]["data"]["parameters"]
left_cam_f_px = left_cam_model["f"]["val"]
left_cam_cx_px = left_cam_model["cx"]["val"]
left_cam_cy_px = left_cam_model["cy"]["val"]
left_cam_f_mm = left_cam_f_px * (sensor_width_mm / sensor_width_px)
left_shift_x = (left_cam_cx_px - sensor_width_px / 2) / sensor_width_px
left_shift_y = (left_cam_cy_px - sensor_height_px /2) / sensor_height_px


right_cam_model = calib["Calibration"]["cameras"][1]["model"]["ptr_wrapper"]["data"]["parameters"]
right_cam_f_px = right_cam_model["f"]["val"]
right_cam_cx_px = right_cam_model["cx"]["val"]
right_cam_cy_px = right_cam_model["cy"]["val"]
right_cam_f_mm = right_cam_f_px * (sensor_width_mm / sensor_width_px)
right_shift_x = (right_cam_cx_px - sensor_width_px / 2) / sensor_width_px
right_shift_y = (right_cam_cy_px - sensor_height_px /2) / sensor_height_px


extrinsics = calib["Calibration"]["cameras"][1]["transform"]
rot = extrinsics["rotation"]
rx = rot['rx']
ry = rot['ry']
rz = rot['rz']
trans = extrinsics["translation"]
tx = trans['x']*1000
ty = trans['y']*1000
tz = trans['z']*1000

print('left camera focal length:', left_cam_f_px)
print('baseline in mm:', tx)

# Load EXR file
file_L = cv.imread("/media/jacob/SNAKE_2TB/Blender_bin/Endo_scans/stereo_test_images/depth_000025_L.exr", cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
file_R = cv.imread("/media/jacob/SNAKE_2TB/Blender_bin/Endo_scans/stereo_test_images/depth_000025_R.exr", cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)

gray_image_L = cv.cvtColor(file_L, cv.COLOR_BGR2GRAY)
gray_image_R = cv.cvtColor(file_R, cv.COLOR_BGR2GRAY)


def dist2depth_img(dist_img, focal =3500):
    img_width = dist_img.shape[1]
    img_height = dist_img.shape[0]

    cx = img_width // 2
    cy = img_height// 2

    xs = np.arange(img_width) - cx
    ys = np.arange(img_height) - cy
    xis, yis = np.meshgrid(xs, ys)
    depth = np.sqrt(dist_img**2 / ((xis**2 + yis**2)/ (focal**2) +1))

    return depth.astype(np.float32)

gray_depth_L = dist2depth_img(gray_image_L)
gray_depth_R = dist2depth_img(gray_image_R)

disp_L = (left_cam_f_px * -tx) / gray_depth_L
print(disp_L)

depth_o3d_L = o3d.geometry.Image(gray_depth_L)
depth_o3d_R = o3d.geometry.Image(gray_depth_R)

pcd_L = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d_L, o3d.camera.PinholeCameraIntrinsic(
            640, 480, 3500, 3500, 340, 240 ))
pcd_R = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d_R, o3d.camera.PinholeCameraIntrinsic(
            640, 480, 3500, 3500, 340, 240 ))
o3d.visualization.draw_geometries([pcd_L, pcd_R])

print(gray_depth_R)
print(gray_depth_R.dtype)
print(gray_depth_R.shape)

(min_val, max_val, min_loc, max_loc) = cv.minMaxLoc(gray_image_R)

normalized_image = cv.normalize(disp_L, None, 0 , 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
print(f"Min pixel value: {min_val} at location {min_loc}")
print(f"Max pixel value: {max_val} at location {max_loc}")

points_3D = cv.reprojectImageTo3D(disp_L, Q, handleMissingValues=False)
print(points_3D.dtype)

