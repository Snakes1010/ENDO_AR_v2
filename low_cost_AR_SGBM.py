import cv2 as cv
import numpy as np
import _torch_scratch
import open3d as o3d
from pytorch3d.io import load_ply

################################################################
# REMAPING
cv_file = cv.FileStorage()
cv_file.open('lowcost_7_25.xml', cv.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
################################################################
# import Q

fs = cv.FileStorage('Q_7_25.yaml', cv.FILE_STORAGE_READ)
Q = fs.getNode('Q').mat()
print("Loaded Q:\n", Q)

# Release the FileStorage object
fs.release()
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

# Initial parameters
min_disp = 2
max_disp = 31
num_disp = max_disp-min_disp
blockSize=7

# Create StereoSGBM object
stereo = cv.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=9,
    P1=8*3*blockSize**2,
    P2=32*3*blockSize**2,
    disp12MaxDiff=1,
    uniquenessRatio=5,
    speckleWindowSize=50,
    speckleRange=2,
    # preFilterCap=63,
    mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
)

# Callback function for the trackbars
def update(val):
    stereo.setMinDisparity(cv.getTrackbarPos('minDisparity', 'Depth'))
    stereo.setNumDisparities(cv.getTrackbarPos('numDisparities', 'Depth'))
    stereo.setBlockSize(cv.getTrackbarPos('blockSize', 'Depth'))
    stereo.setP1(cv.getTrackbarPos('P1', 'Depth'))
    stereo.setP2(cv.getTrackbarPos('P2', 'Depth'))
    stereo.setDisp12MaxDiff(cv.getTrackbarPos('disp12MaxDiff', 'Depth'))
    stereo.setUniquenessRatio(cv.getTrackbarPos('uniquenessRatio', 'Depth'))
    stereo.setSpeckleWindowSize(cv.getTrackbarPos('speckleWindowSize', 'Depth'))
    stereo.setSpeckleRange(cv.getTrackbarPos('speckleRange', 'Depth'))
    # stereo.setPreFilterCap(cv.getTrackbarPos('preFilterCap', 'Depth'))

# Create trackbars
cv.namedWindow('Depth')
cv.createTrackbar('minDisparity', 'Depth', min_disp, 125, update)
cv.createTrackbar('numDisparities', 'Depth', num_disp, 200, update)
cv.createTrackbar('blockSize', 'Depth', 7, 30, update)
cv.createTrackbar('P1', 'Depth', blockSize, 50, update)
cv.createTrackbar('P2', 'Depth', blockSize, 50, update)
cv.createTrackbar('disp12MaxDiff', 'Depth', 1, 100, update)
cv.createTrackbar('uniquenessRatio', 'Depth', 0, 100, update)
cv.createTrackbar('speckleWindowSize', 'Depth', 0, 200, update)
cv.createTrackbar('speckleRange', 'Depth', 2, 5, update)
# cv.createTrackbar('preFilterCap', 'Depth', 56, 100, update)

########################################################################
#pointcloud ply

# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
#     print('GPU enabled')
# else:
#     device = torch.device("cpu")
#     print("WARNING: CPU only, this will be slow!")
cloudnumber =0

def createPointCloudFileColor(vertices, colors, filename):
    vertices = vertices.reshape(-1, 3)
    colors = colors.reshape(-1, 3)

    data = np.hstack([vertices, colors]).astype('float32')

    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(data)))
        np.savetxt(f, data, '%f %f %f %d %d %d')


output_file = 'pointcloud.ply'


def createPointCloudFile(vertices, filename):
    vertices = np.hstack([vertices.reshape(-1,3), colors])

    ply_header = '''
    ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    with open(filename, 'w') as f:
        f.write(ply_header %dict(vert_num=len(vertices)))
        np.savetxt(f,vertices, '%f %f %f %d %d %d')
output_file = 'pointcloud.ply'

def create_point_cloud(vertices, colors):
    # Reshape color and concatenate with vertices
    colors = colors.reshape(-1,3)
    vertices = np.hstack([vertices.reshape(-1,3), colors])

    # Create a structured array
    point_cloud = np.empty(len(vertices), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    # Fill structured array
    point_cloud['x'] = vertices[:,0]
    point_cloud['y'] = vertices[:,1]
    point_cloud['z'] = vertices[:,2]
    point_cloud['red'] = colors[:,0]
    point_cloud['green'] = colors[:,1]
    point_cloud['blue'] = colors[:,2]

    return point_cloud

################################################################
# Overlay
alpha_overlay = 50
beta_overlay = 50
gama_overlay = 0
cv.namedWindow("Overlay")
def update_overlay_alpha(val):
    global alpha_overlay
    alpha_overlay  = val/1000
def update_overlay_beta(val):
    global beta_overlay
    beta_overlay  = val/1000
def update_overlay_gamma(val):
    global gamma_overlay
    gamma_overlay  = val

# Create trackbars
cv.createTrackbar('alpha', 'Overlay', alpha_overlay, 1000, update_overlay_alpha)
cv.createTrackbar('beta', 'Overlay', beta_overlay, 10000, update_overlay_beta)
cv.createTrackbar('gamma', 'Overlay', gama_overlay, 10, update_overlay_alpha)

#################################################################
#CAMERA SET UP

left_cam = cv.VideoCapture(0)
right_cam = cv.VideoCapture(2)

print_shape_once = True

left_cam.set(cv.CAP_PROP_BRIGHTNESS, 0)  # Brightness: -64 to 64, default: 0
left_cam.set(cv.CAP_PROP_CONTRAST, 32)  # Contrast: 0 to 64, default: 32
left_cam.set(cv.CAP_PROP_SATURATION, 64)  # Saturation: 0 to 128, default: 64
left_cam.set(cv.CAP_PROP_HUE, 0)  # Hue: -40 to 40, default: 0
left_cam.set(cv.CAP_PROP_AUTO_WB, 1)  # White Balance, Automatic: 0 or 1, default: 1
left_cam.set(cv.CAP_PROP_GAMMA, 100)  # Gamma: 72 to 500, default: 100
left_cam.set(cv.CAP_PROP_GAIN, 0)  # Gain: 0 to 100, default: 0
left_cam.set(cv.CAP_PROP_SHARPNESS, 2)  # Sharpness: 0 to 6, default: 2
left_cam.set(cv.CAP_PROP_BACKLIGHT, 1)  # Backlight Compensation: 0 to 4, default: 1
left_cam.set(cv.CAP_PROP_AUTO_EXPOSURE, 3)  # Auto Exposure: 0 to 3, default: 1
left_cam.set(cv.CAP_PROP_EXPOSURE, 250) # Exposure Time, Absolute: 1 to 5000, default: 473
left_cam.set(cv.CAP_PROP_AUTO_EXPOSURE, 0)  # Exposure, Dynamic Framerate: 0 or 1, default: 0

right_cam.set(cv.CAP_PROP_BRIGHTNESS, 0)  # Brightness: -64 to 64, default: 0
right_cam.set(cv.CAP_PROP_CONTRAST, 32)  # Contrast: 0 to 64, default: 32
right_cam.set(cv.CAP_PROP_SATURATION, 64)  # Saturation: 0 to 128, default: 64
right_cam.set(cv.CAP_PROP_HUE, 0)  # Hue: -40 to 40, default: 0
right_cam.set(cv.CAP_PROP_AUTO_WB, 1)  # White Balance, Automatic: 0 or 1, default: 1
right_cam.set(cv.CAP_PROP_GAMMA, 100)  # Gamma: 72 to 500, default: 100
right_cam.set(cv.CAP_PROP_GAIN, 0)  # Gain: 0 to 100, default: 0
right_cam.set(cv.CAP_PROP_SHARPNESS, 2)  # Sharpness: 0 to 6, default: 2
right_cam.set(cv.CAP_PROP_BACKLIGHT, 1)  # Backlight Compensation: 0 to 4, default: 1
right_cam.set(cv.CAP_PROP_AUTO_EXPOSURE, 3)  # Auto Exposure: 0 to 3, default: 1
right_cam.set(cv.CAP_PROP_EXPOSURE, 250)  # Exposure Time, Absolute: 1 to 5000, default: 473
right_cam.set(cv.CAP_PROP_AUTO_EXPOSURE, 0)  # Exposure, Dynamic Framerate: 0 or 1, default: 0

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

        # cv.line(resized, (0,(360)), (1920, 360),(255,0,255), 3)
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
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(output_points)
            # pcd.colors = o3d.utility.Vector3dVector(output_colors)
            aabb = pcd.get_axis_aligned_bounding_box()
            # Get an oriented bounding box (OBB)
            obb = pcd.get_oriented_bounding_box()
            aabb.color = (1, 0, 0)  # red

            # Set color and line width for OBB
            obb.color = (0, 1, 0)  # green

            # Visualize the point cloud
            o3d.visualization.draw_geometries([pcd, aabb, obb])

        if cv.waitKey(1) & 0xFF == ord('q'):
                break

left_cam.release()
right_cam.release()
cv.destroyAllWindows()
