import numpy as np
import cv2 as cv
import open3d as o3d
from open3d import camera
import AR_functions

print("testing camera in opend3d...")
K, k, cam_rvecs, cam_tvecs = AR_functions.readCalibParameters('/home/jacob/endo_calib/low_cost_proj/8_11_2x/low_cost_dual_Charuco.json')
camera_matrix_L = np.array(K[0])
camera_matrix_R = np.array(K[1])
distL = np.array(k[0][:4])
distR = np.array(k[1][:4])
r_vecs = cam_rvecs[1]
t_vecs = cam_tvecs[1]
R,_ = cv.Rodrigues(r_vecs)

intrinsic_L = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, fx=camera_matrix_L[0,0], fy=camera_matrix_L[1,1], cx=camera_matrix_L[0,2], cy=camera_matrix_L[1,2])
intrinsic_R = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, fx=camera_matrix_R[0,0], fy=camera_matrix_R[1,1], cx=camera_matrix_R[0,2], cy=camera_matrix_R[1,2])

extrinsics = np.eye(4)
extrinsics[:3, :3] = R
extrinsics[:3, 3] = t_vecs.squeeze()

camera_L = o3d.camera.PinholeCameraParameters()
camera_L.intrinsic = intrinsic_L
camera_L.extrinsic = np.eye(4)
# print(camera_L.intrinsic)
# print(intrinsic_L.intrinsic_matrix)
# print(camera_L.extrinsic)

camera_R = o3d.camera.PinholeCameraParameters()
camera_R.intrinsic = intrinsic_R
camera_R.extrinsic = extrinsics
# print(camera_R.extrinsic)


mesh = o3d.io.read_triangle_mesh("J.ply")

# Create visualizer and add the mesh and the cameras
vis_L = o3d.visualization.Visualizer()
vis_R = o3d.visualization.Visualizer()
vis_L.create_window(window_name='VIEWS LEFT', width=640, height=480)
vis_R.create_window(window_name='VIEWS RIGHT', width=640, height=480)
vis_L.add_geometry(mesh)
vis_R.add_geometry(mesh)
ctr_L = vis_L.get_view_control()
ctr_R = vis_R.get_view_control()
start = ctr_L.convert_to_pinhole_camera_parameters()

print("Start:", start.intrinsic)
print('default starting int camera matrix: \n', start.intrinsic.intrinsic_matrix)
print('default starting ext camera matrix: \n', start.extrinsic)

# Render the view from the first camera
left = ctr_L.convert_from_pinhole_camera_parameters(camera_L)
print(left)
vis_L.run()  # Manually close the window to continue to the next view

# Render the view from the second camera
ctr_R.convert_from_pinhole_camera_parameters(camera_R)
vis_R.run()  # Manually close the window to end the program

vis_L.destroy_window()
vis_R.destroy_window()