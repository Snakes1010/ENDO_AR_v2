import numpy as np
import AR_functions


K_L, k_L, cam_rvecs_L, cam_tvecs_L = AR_functions.readCalibParameters('/home/jacob/endo_calib/camera_calibration_5_26/8_11_5_26/left_cal_single.json')
K_R, k_R, cam_rvecs_R, cam_tvecs_R = AR_functions.readCalibParameters('/home/jacob/endo_calib/camera_calibration_5_26/8_11_5_26/right_cal_single.json')

camera_matrix_L = np.array(K_L[0])
camera_matrix_R = np.array(K_R[0])




print('camera_matrix_L:\n', camera_matrix_L)
print('camera_matrix_R:\n', camera_matrix_R)


################################3333
# intrinsics = jsonStruct["Calibration"]["cameras"][i]["model"]["ptr_wrapper"]["data"]["parameters"]
# f = intrinsics["f"]["val"]
# ar = intrinsics["ar"]["val"]
# cx = intrinsics["cx"]["val"]
# cy = intrinsics["cy"]["val"]
# k1 = intrinsics["k1"]["val"]
# k2 = intrinsics["k2"]["val"]
# k3 = intrinsics["k3"]["val"]
# k4 = intrinsics["k4"]["val"]
# k5 = intrinsics["k5"]["val"]
# k6 = intrinsics["k6"]["val"]
# p1 = intrinsics["p1"]["val"]
# p2 = intrinsics["p2"]["val"]
# s1 = intrinsics["s1"]["val"]
# s2 = intrinsics["s2"]["val"]
# s3 = intrinsics["s3"]["val"]
# s4 = intrinsics["s4"]["val"]
# tauX = intrinsics["tauX"]["val"]
# tauY = intrinsics["tauY"]["val"]
#
# K.append(np.array([[f, 0.0, cx], [0.0, f * ar, cy], [0.0, 0.0, 1.0]], dtype=np.float64))
# k.append(np.array([k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tauX, tauY], dtype=np.float64))
#
# transform = jsonStruct["Calibration"]["cameras"][i]["transform"]
# rot = transform["rotation"]
# cam_rvecs.append(np.array([rot["rx"], rot["ry"], rot["rz"]], dtype=np.float64))
# t = transform["translation"]
# cam_tvecs.append(np.array([t["x"], t["y"], t["z"]], dtype=np.float64))