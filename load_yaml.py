import numpy as np
import yaml

camera_parameters_L_file ='calib_100_L3.yaml'
camera_parameters_R_file ='calib_100_R3.yaml'

with open(camera_parameters_L_file, 'r') as file:
    params_dict_L = yaml.load(file, Loader=yaml.FullLoader)

    camera_matrix_L = np.array(params_dict_L['camera_matrix_L'])
    dist_coeffs_L = np.array(params_dict_L['dist_coeffs_L'])
    rms_L = params_dict_L['rms_L']
    rvecs_L = np.array(params_dict_L['rvecs_L'])
    tvesc_L = np.array(params_dict_L['tvecs_L'])

with open(camera_parameters_R_file, 'r') as file:
    params_dict_R = yaml.load(file, Loader=yaml.FullLoader)

    camera_matrix_R = np.array(params_dict_R['camera_matrix_R'])
    dist_coeffs_R = np.array(params_dict_R['dist_coeffs_R'])
    rms_R = np.array(params_dict_R['rms_R'])
    rvecs_R = np.array(params_dict_R['rvecs_R'])
    tvesc_R = np.array(params_dict_R['tvecs_R'])

# view the dictionary of parameters
print("camera_matrix_L: \n", camera_matrix_L, "\n")
print("dist_coeffs_L: \n", dist_coeffs_L, "\n")
print("rms_L: \n", rms_L, "\n")
print("rvecs_L: \n", rvecs_L, "\n")
print("tvesc_L: \n", tvesc_L, "\n")
print("camera_matrix_R: \n", camera_matrix_R, "\n")
print("dist_coeffs_R: \n", dist_coeffs_R, "\n")
print("rms_R: \n", rms_R, "\n")
print("rvecs_R: \n", rvecs_R, "\n")
print("tvesc_R: \n", tvesc_R, "\n")
