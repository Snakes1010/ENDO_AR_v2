import cv2 as cv
import numpy as np
import _torch_scratch
import open3d as o3d
import torchvision.transforms as transforms
from PIL import Image
from monodepth2.networks.dispnet import DispNet
from pytorch3d.io import load_ply

################################################################
# REMAPING
cv_file = cv.FileStorage()
cv_file.open('lowcost_7_14.xml', cv.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
################################################################
# import Q

fs = cv.FileStorage('Q_7_14.yaml', cv.FILE_STORAGE_READ)
Q = fs.getNode('Q').mat()
print("Loaded Q:\n", Q)

# Release the FileStorage object
fs.release()


#
#pointcloud ply

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('GPU enabled')
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")
cloudnumber =0





# Create trackbars
cv.createTrackbar('alpha', 'Overlay', alpha_overlay, 1000, update_overlay_alpha)
cv.createTrackbar('beta', 'Overlay', beta_overlay, 10000, update_overlay_beta)
cv.createTrackbar('gamma', 'Overlay', gama_overlay, 10, update_overlay_alpha)

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
    depth = stereo.compute(gray_left_remap,gray_right_remap)


    if AR_use == True:
        combined_rectify = np.concatenate((frame_left_remap, frame_right_remap), axis=1)
        resized = cv.resize(combined_rectify, dim, interpolation=cv.INTER_AREA)

        cv.line(resized, (0,(360)), (1920, 360),(255,0,255), 3)
        cv.imshow('AR', resized)
        cv.moveWindow('AR', 5600, 150)
        cv.imshow('Depth', depth/2000)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        combined_rectify = np.concatenate((frame_left_remap, frame_right_remap), axis=1)
        overlay = cv.addWeighted(gray_left_remap, alpha_overlay, gray_right_remap, beta_overlay, gama_overlay)
        cv.imshow("Overlay", overlay)
        cv.imshow('AR', combined_rectify)
        cv.imshow('Depth', depth/2500)
        disparity_normalized = np.zeros(depth.shape, np.uint8)
        cv.normalize(depth, disparity_normalized, alpha=255, beta=0, dtype=cv.CV_8U, norm_type=cv.NORM_MINMAX)
        colored_disparity_map = cv.applyColorMap(disparity_normalized, cv.COLORMAP_JET)
        cv.imshow('Colored Disparity Map', colored_disparity_map)

        # point cloud operations
        if cv.waitKey(1) & 0xFF == ord('p'):


            cloudnumber += 1
            output_file = (f"pointcloud{cloudnumber}.ply")
            print(f"pointcloud{cloudnumber}.ply")
            disparity_map = np.float32(np.divide(depth,16.0))
            print(np.min(disparity_map))
            print(np.max(disparity_map))
            points_3D = cv.reprojectImageTo3D(disparity_map, Q, handleMissingValues=False)
            colors = cv.cvtColor(frame_L, cv.COLOR_BGR2RGB)
            mask_map = disparity_map > disparity_map.min()
            output_points = points_3D[mask_map]
            output_colors = colors[mask_map]
            point_cloud = createPointCloudFileColor(output_points, output_colors,output_file)
            # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            #     size=.01, origin=[0, 0, 1])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(output_points)
            # pcd.colors = o3d.utility.Vector3dVector(output_colors)

            # Visualize the point cloud
            o3d.visualization.draw_geometries([pcd])

        if cv.waitKey(1) & 0xFF == ord('q'):
                break

left_cam.release()
right_cam.release()
cv.destroyAllWindows()
