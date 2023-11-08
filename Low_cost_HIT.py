import cv2 as cv
import numpy as np
import _torch_scratch
import open3d as o3d
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

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('GPU enabled')
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")
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
