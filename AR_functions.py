import cv2 as cv
import numpy as np
import os
import random
import time
import yaml
import glob
import json

# for an individual camera, get the fast checkerboards and but the
# objpoints and imgpoints into lists

def calibrate_fine(images, chessboardSize):

    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    xv, yv = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]]
    objp[:, :2] = np.array([yv, xv]).T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []
    imgloop = 0
    total_time = 0

    for img in images:
        imgloop += 1
        start_time = time.time()
        img = cv.imread(img)
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCornersSB(gray_img, chessboardSize, flags= None)
        if ret == True:
            corners = orientation(corners)
            objpoints.append(objp)
            imgpoints.append(corners)
            cv.drawChessboardCorners(img, chessboardSize, corners, ret)
            cv.putText(img, str(imgloop), (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            draw_first_point(img, corners)
            draw_last_point(img, corners)
            # draw_second_point(img, corners, objp)
            cv.imshow('img ' + os.path.basename(images[0]), img)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                cv.destroyAllWindows()
                break
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time

    print(f"Total processing time: {total_time:.4f} seconds")
    return objpoints, imgpoints

################################################################
def draw_first_point(image, corners, color=(255, 255, 0), radius=5):
    if len(corners) > 0:
        first_point = tuple(corners[0][0].astype(int))
        cv.circle(image, first_point, radius, color, thickness=-1)
        coordinates_text = f"({first_point[0]}, {first_point[1]}, F)"
        text_offset = (first_point[0] - 250, first_point[1])
        cv.putText(image, coordinates_text, text_offset, cv.FONT_HERSHEY_SIMPLEX, 1, color, thickness=3)
def draw_last_point(image, corners, color=(0, 0, 255), radius=5):
    if len(corners) > 0:
        last_point = tuple(corners[-1][-1].astype(int))
        cv.circle(image, last_point, radius, color, thickness=-1)
        coordinates_text = f"({last_point[0]}, {last_point[1]}, L)"
        text_offset = (last_point[0] + 25, last_point[1])
        cv.putText(image, coordinates_text, text_offset, cv.FONT_HERSHEY_SIMPLEX, 1, color, thickness=3)

#############################################################
def choose_stereo_pairs(images_L, images_R, chessboardSize):
    def on_trackbar(pos):
        pass

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv.namedWindow('combined')
    cv.createTrackbar('Frame', 'combined', 0, len(images_L) - 1, on_trackbar)
    left_chosen_path = []
    right_chosen_path = []

    while True:
        pos = cv.getTrackbarPos('Frame', 'combined')
        img_l = cv.imread(images_L[pos])
        img_r= cv.imread(images_R[pos])
        grayL = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)

        retL, cornersL = cv.findChessboardCornersSB(grayL, chessboardSize, flags= None)
        retR, cornersR = cv.findChessboardCornersSB(grayR, chessboardSize, flags= None)

        if retL and retR == True:
            cornersL = orientation(cornersL)
            cornersR = orientation(cornersR)
            cv.drawChessboardCorners(img_l, chessboardSize, cornersL, retL)
            cv.drawChessboardCorners(img_r, chessboardSize, cornersR, retR)
            draw_first_point(img_l, cornersL)
            draw_first_point(img_r, cornersR)
            draw_last_point(img_l, cornersL)
            draw_last_point(img_r, cornersR)
            combined = np.concatenate((img_l, img_r), axis=1)
            cv.putText(combined, str(pos), (10, 50), cv.FONT_ITALIC, 2, (255, 255, 255), 2, cv.LINE_AA)
        else:
            combined = np.concatenate((img_l, img_r), axis=1)
            cv.putText(combined, "NO MATCH FOUND_" + str(pos), (10, 50), cv.FONT_ITALIC, 2, (0, 0, 255), 4, cv.LINE_AA)

        cv.imshow("combined", combined)
        key = cv.waitKey(10) & 0xFF

        if key == ord('q'):
            cv.destroyAllWindows()
            break
        if key == ord('s'):
            left_chosen_path.append(images_L[pos])
            right_chosen_path.append(images_R[pos])
            print(left_chosen_path)
            print(right_chosen_path)
            print("Image Added #_" + str(len(left_chosen_path)))

    return left_chosen_path, right_chosen_path
############################################
def random_imgs_from_lst(image_lst, obj_lst,  num_items):
    # check if the list is empty or the number of items to extract is greater than the list length
    if len(image_lst) == 0 or num_items > len(image_lst):
        return []
    # extract the selected number of items from the list at random
    random_imgs = random.sample(image_lst, num_items)
    random_objs = random.sample(obj_lst, num_items)

    return random_imgs, random_objs
##############################################
def stereo_calibrate(objpoints,imgpointsL, imgpointsR, new_camera_martix_L, distL, new_camera_matrix_R, distR, width, height):
    flags = 0
    flags |= cv.CALIB_USE_INTRINSIC_GUESS | cv.CALIB_FIX_INTRINSIC | cv.CALIB_FIX_PRINCIPAL_POINT
    # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
    # Hence intrinsic parameters are the same
    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # This step is performed to transformation between the two cameras and calculate Essential and Fundamental matrix
    # Size of the image used only to initialize the intrinsic camera matrix [w,h].
    retStereo, StereoCameraMatrixL, distL_stereo, StereoCameraMatrixR, distR_stereo, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR,
                                                            new_camera_martix_L, distL, new_camera_matrix_R, distR,(width, height), criteria_stereo, flags)
    stereo_calib_params = {'retStereo': retStereo,
                           'StereoCameraMatrixL': StereoCameraMatrixL.tolist(),
                           'distL_stereo': distL_stereo.tolist(),
                           'StereoCameraMatrixR': StereoCameraMatrixR.tolist(),
                           'distR_stereo': distR_stereo.tolist(),
                           'rot': rot.tolist(),
                           'trans': trans.tolist(),
                           'essentialMatrix': essentialMatrix.tolist(),
                           'fundamentalMatrix': fundamentalMatrix.tolist()}

    # save the dictionary to a YAML file
    with open('stereo_calib_params.yml', 'w') as file:
        yaml.dump(stereo_calib_params, file)

    return retStereo, StereoCameraMatrixL, distL_stereo, StereoCameraMatrixR, distR_stereo, rot, trans, essentialMatrix, fundamentalMatrix

######################

def rectify_undistort(StereoCameraMatrixL,distL_stereo, StereoCameraMatrixR, distR_stereo,img_size_w_h, rot, trans):
    # StereoVison rectification
    flags = cv.CALIB_ZERO_DISPARITY
    alpha = 0.9
    rectL_stereo, rectR_stereo, projMatrixL, projMatrixR, Q, roi_L2, roi_R2 = cv.stereoRectify(
        StereoCameraMatrixL,distL_stereo, StereoCameraMatrixR, distR_stereo, img_size_w_h, rot, trans, flags, alpha)

    stereoMapL_x, stereoMapL_y = cv.initUndistortRectifyMap(StereoCameraMatrixL, distL_stereo, rectL_stereo, projMatrixL,
                                            img_size_w_h, cv.CV_32FC2)
    stereoMapR_x, stereoMapR_y = cv.initUndistortRectifyMap(StereoCameraMatrixR, distR_stereo, rectR_stereo, projMatrixR,
                                            img_size_w_h, cv.CV_32FC2)

    print("Saving parameters!")
    cv_file = cv.FileStorage('stereo_map.xml', cv.FILE_STORAGE_WRITE)
    cv_file.write('stereoMapL_x', stereoMapL_x)
    cv_file.write('stereoMapL_y', stereoMapL_y)
    cv_file.write('stereoMapR_x', stereoMapR_x)
    cv_file.write('stereoMapR_y', stereoMapR_y)
    cv_file.release()
    print("Saved!")

    return stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y

################################################################
def load_yaml_calibration(camera_parameters_L_file, camera_parameters_R_file):
    if os.path.exists(camera_parameters_L_file) or os.path.exists(camera_parameters_R_file):
        with open(camera_parameters_L_file, 'r') as file:
            params_dict_L = yaml.load(file, Loader=yaml.FullLoader)

        camera_matrix_L = np.array(params_dict_L['camera_matrix_L'])
        dist_coeffs_L = np.array(params_dict_L['dist_coeffs_L'])
        rms_L = params_dict_L['rms_L']
        rvecs_L = np.array(params_dict_L['rvecs_L'])
        tvecs_L = np.array(params_dict_L['tvecs_L'])

        with open(camera_parameters_R_file, 'r') as file:
            params_dict_R = yaml.load(file, Loader=yaml.FullLoader)

        camera_matrix_R = np.array(params_dict_R['camera_matrix_R'])
        dist_coeffs_R = np.array(params_dict_R['dist_coeffs_R'])
        rms_R = np.array(params_dict_R['rms_R'])
        rvecs_R = np.array(params_dict_R['rvecs_R'])
        tvecs_R = np.array(params_dict_R['tvecs_R'])
    else:
        print("No camera parameters found")
def save_yaml_file(filename, data):
    with open(filename, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

################################################################
def calibrate_fast(images, chessboardSize):

    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    xv, yv = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]]
    objp[:, :2] = np.array([yv, xv]).T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []
    imgloop = 0
    total_time = 0

    for img in images:
        imgloop += 1
        start_time = time.time()
        img = cv.imread(img)
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray_img, chessboardSize, None, flags= cv.CALIB_CB_FAST_CHECK)
        if ret == True:
            objpoints.append(objp)
            corners = corners[::-1]
            imgpoints.append(corners)
            cv.drawChessboardCorners(img, chessboardSize, corners, ret)
            cv.putText(img, str(imgloop), (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv.imshow('img ' + os.path.basename(images[0]), img)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                cv.destroyAllWindows()
                break
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time

    print(f"Total processing time: {total_time:.4f} seconds")
    return objpoints, imgpoints

def orientation(corners):
        first_point = tuple(corners[0][0].astype(int))
        last_point = tuple(corners[-1][-1].astype(int))
        if first_point[0] > last_point[0]:
            corners = corners[::-1]
            return corners
        else:
            return corners

################################################################
def import_yaml(file_name):
    with open(file_name, 'r') as file:
        data = yaml.safe_load(file)
    return data

################################################################
def export_yaml(file_name, data):
    with open(file_name, 'w') as file:
        yaml.dump(data, file)

################################################################
def readCalibParameters(filePath):
    with open(filePath) as file:
        jsonStruct = json.load(file)

    assert jsonStruct["Calibration"]["cameras"][0]["model"]["polymorphic_name"] == "libCalib::CameraModelOpenCV"

    nCameras = len(jsonStruct["Calibration"]["cameras"])
    if nCameras < 1:
        return False

    K = []
    k = []
    cam_rvecs = []
    cam_tvecs = []

    for i in range(nCameras):
        intrinsics = jsonStruct["Calibration"]["cameras"][i]["model"]["ptr_wrapper"]["data"]["parameters"]
        f = intrinsics["f"]["val"]
        ar = intrinsics["ar"]["val"]
        cx = intrinsics["cx"]["val"]
        cy = intrinsics["cy"]["val"]
        k1 = intrinsics["k1"]["val"]
        k2 = intrinsics["k2"]["val"]
        k3 = intrinsics["k3"]["val"]
        k4 = intrinsics["k4"]["val"]
        k5 = intrinsics["k5"]["val"]
        k6 = intrinsics["k6"]["val"]
        p1 = intrinsics["p1"]["val"]
        p2 = intrinsics["p2"]["val"]
        s1 = intrinsics["s1"]["val"]
        s2 = intrinsics["s2"]["val"]
        s3 = intrinsics["s3"]["val"]
        s4 = intrinsics["s4"]["val"]
        tauX = intrinsics["tauX"]["val"]
        tauY = intrinsics["tauY"]["val"]

        K.append(np.array([[f, 0.0, cx], [0.0, f * ar, cy], [0.0, 0.0, 1.0]], dtype=np.float64))
        k.append(np.array([k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tauX, tauY], dtype=np.float64))

        transform = jsonStruct["Calibration"]["cameras"][i]["transform"]
        rot = transform["rotation"]
        cam_rvecs.append(np.array([rot["rx"], rot["ry"], rot["rz"]], dtype=np.float64))
        t = transform["translation"]
        cam_tvecs.append(np.array([t["x"], t["y"], t["z"]], dtype=np.float64))

    return K, k, cam_rvecs, cam_tvecs