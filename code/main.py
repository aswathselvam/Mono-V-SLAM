from load_dataset import get_image_paths, get_poses, get_K
from compute_pose import *
from map import Map

img_paths = get_image_paths()
gt_poses = get_poses()
K = get_K()
map = Map()

for pose, pointclouds in compute_camera_pose(img_paths, K):
    print(pose)

    # Update data in the Map
    
    
    # Choose Keyframes


    # Perform BA


    # Detect and perform loop closure


    # Plot the Map