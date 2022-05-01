import os 
import pandas as pd
import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt


dataset_folder = os.environ["SLAM_DATA_FOLDER"]
# print(dataset_folder)
image_folder = os.path.join(dataset_folder,"sequences","00","image_0/")

def get_image_paths() -> list:
    frames = []
    imgs = []
    for frame in sorted(glob(image_folder+'*.png')):
        # img=cv2.imread(frame,0)
        # imgs.append(img)
        frames.append(frame)
    # frames.sort()
    # print(frames)
    # cv2.imshow("ef",imgs[0])
    # cv2.waitKey(0)
    return frames


def get_poses() -> np.array([...,3,4]):
    poses_file = os.path.join(dataset_folder,"poses","00.txt")
    poses = pd.read_csv(poses_file, delimiter=' ', header=None)

    gt_poses = np.zeros((len(poses), 3, 4))
    for i in range(len(poses)):
        gt_poses[i] = np.array(poses.iloc[i]).reshape((3, 4))

    
    # fig = plt.figure(figsize=(5,5))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(gt_poses[:, :, 3][:, 0], gt_poses[:, :, 3][:, 1], gt_poses[:, :, 3][:, 2])
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.view_init(elev=-40, azim=270)
    # plt.show()

    return gt_poses

def get_K() -> np.array([3,4]):
    camera_params = pd.read_csv(os.path.join(dataset_folder,"sequences","00",'calib.txt'), delimiter=' ', header=None, index_col=0)
    P0 = np.array(camera_params.loc['P0:']).reshape((3,4))
    # print(P0)
    return P0
    
# get_image_paths()
# get_poses()
# get_K()