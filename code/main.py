from load_dataset import get_image_paths, get_ground_truth_poses, get_K
from compute_pose import *
from map import Map
from match import *
import cv2
from GetInliersRANSAC import *
from EssentialMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from visualization_utils import *
from DisambiguateCameraPose import *
from plotting import *
from ego_state import State

def H_matrix(rot_mat, t):
    i = np.column_stack((rot_mat, t))
    a = np.array([0, 0, 0, 1])
    H = np.vstack((i, a))
    return H

img_paths = get_image_paths()
gt_poses = get_ground_truth_poses()
K = get_K()
map = Map()
state = State(x=0,y=0,z=0)
plotter = Plot()
frame_num = 0


for pose, pointclouds in compute_camera_pose(img_paths, K):

    for i in range(len(pointclouds[0])):
        pointclouds[i] = pointclouds[i]+pose[:3,3]
        
    pointcloud = np.asarray(pointclouds)
    new_pose = State(pose[0,3], pose[1,3], pose[2,3])
    state += new_pose
    plotter.plot_trajectory(state, gt_poses[:frame_num])
    frame_num += 1

    # Update data in the Map
    
    
    # Choose Keyframes


    # Perform BA


    # Detect and perform loop closure


    # Plot the Map