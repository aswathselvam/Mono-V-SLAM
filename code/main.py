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
import copy

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
# plotter = Plot()
pangolinplotter = PangolinPlot()
frame_num = 1

import time 

for image, image_points, pose, pointcloud in compute_camera_pose(img_paths, K):

    start = time.time()

    # for i in range(len(pointcloud)):
    #     # pointcloud[i] = pointcloud[i]+gt_poses[frame_num,:,3]
    #     pointcloud[i].update(gt_poses[frame_num,:,:])
        
    # pointcloud = np.asarray(pointcloud)
    # pangolinplotter.plot_point_cloud(pointcloud[-30:])
    state.update(pose)
    # pangolinplotter.plot_trajectory(state, gt_poses[:frame_num])
    frame_num += 1
    # print("Time:", start - time.time())

    # Update data in the Map
    map.update(image, image_points, copy.deepcopy(state), gt_poses[frame_num,:,3], pointcloud)
    
    
    # Choose Keyframes


    # Perform BA


    # Detect and perform loop closure


    # Plot the Map
    pangolinplotter.draw(map)