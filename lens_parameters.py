import yaml
import os
import numpy as np

camera_parameters_L_file ='calib_100_L.yaml'
camera_parameters_R_file ='calib_100_R.yaml'

if os.path.exists(camera_parameters_L_file) or os.path.exists(camera_parameters_R_file):
    with open(camera_parameters_L_file, 'r') as file:
        params_dict_L = yaml.load(file, Loader=yaml.FullLoader)
    new_camera_matrix_L = np.array(params_dict_L['new_camera_matrix_L'])
    camera_matrix_L = np.array(params_dict_L['camera_matrix_L'])
    dist_coeffs_L = np.array(params_dict_L['dist_coeffs_L'])
    rms_L = params_dict_L['rms_L']
    rvecs_L = np.array(params_dict_L['rvecs_L'])
    tvecs_L = np.array(params_dict_L['tvecs_L'])

    with open(camera_parameters_R_file, 'r') as file:
        params_dict_R = yaml.load(file, Loader=yaml.FullLoader)

    new_camera_matrix_R = np.array(params_dict_R['new_camera_matrix_R'])
    camera_matrix_R = np.array(params_dict_R['camera_matrix_R'])
    dist_coeffs_R = np.array(params_dict_R['dist_coeffs_R'])
    rms_R = np.array(params_dict_R['rms_R'])
    rvecs_R = np.array(params_dict_R['rvecs_R'])
    tvecs_R = np.array(params_dict_R['tvecs_R'])
else:
    print("No camera parameters file found \n Running calibration...")


pixel_size_mm = 1.85 / 1000

sensor_width_mm = 4096 * pixel_size_mm
sensor_height_mm = 2160 * pixel_size_mm

print('sensor height mm:', sensor_height_mm)
print('sensor width mm:', sensor_width_mm)

sensor_width_px = 4096
sensor_height_px = 2160
image_width_px = 1920
image_height_px = 1080

fx_pixels_L = camera_matrix_L[0,0]
fy_pixels_L = camera_matrix_L[1,1]
new_fx_pixels_L = new_camera_matrix_L[0,0]
new_fy_pixels_L = new_camera_matrix_L[1,1]
fx_pixels_R = camera_matrix_R[0,0]
fy_pixels_R = camera_matrix_L[1,1]
new_fx_pixels_R = new_camera_matrix_R[0,0]
new_fy_pixels_R = new_camera_matrix_R[1,1]


print('camera_matix_L' , camera_matrix_L[1,1])
print(new_camera_matrix_L[0,0])
width_ratio = sensor_width_px/image_width_px
height_ratio = sensor_height_px/image_height_px

effective_pixel_width = pixel_size_mm * width_ratio
effective_pixel_height = pixel_size_mm * height_ratio

print('effective pixel width:', effective_pixel_width)
print('effective pixel height:', effective_pixel_height)

print('width ratio:', width_ratio)
print('height ratio:', height_ratio)

fx_mm_L = fx_pixels_L * effective_pixel_width
fy_mm_L = fy_pixels_L * effective_pixel_height
new_fx_mm_L = new_fx_pixels_L * effective_pixel_width
new_fy_mm_L = new_fy_pixels_L * effective_pixel_height
fx_mm_R = fx_pixels_R * effective_pixel_width
fy_mm_R = fy_pixels_R * effective_pixel_height
new_fx_mm_R = new_fx_pixels_R * effective_pixel_width
new_fy_mm_R = new_fy_pixels_R * effective_pixel_height

print("Left camera focal lengths (fx_mm, fy_mm):", fx_mm_L, fy_mm_L)
print("Right camera focal lengths (fx_mm, fy_mm):", fx_mm_R, fy_mm_R)
print("Left camera new focal lengths (new_fx_mm, new_fy_mm):", new_fx_mm_L, new_fy_mm_L)
print("Right camera new focal lengths (new_fx_mm, new_fy_mm):", new_fx_mm_R, new_fy_mm_R)
