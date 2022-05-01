import numpy as np
from typing import Union,TypeVar
from feature_point import FeaturePoint

def compute_camera_pose(img_paths, K):
    pose = np.zeros([3,4])
    pointclouds = FeaturePoint( np.zeros([3,3]))


    

    yield pose,pointclouds