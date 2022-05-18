# from plotting import plotpointcloud
from frame import Frame
import cv2 
import numpy as np
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

class Map():
    def __init__(self):
        self.pointcloud = None
        self.frames = []
        self.frame_id = 0
        self.old_features = None
        self.prev_keyframe_desc = None

    def update(self,image,image_coordinates,state,gt_poses,pointcloud):
        if self.old_features is None:
            self.old_features = pointcloud[2]
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        
        old, current = len(self.old_features), len(pointcloud[2])
        num = current if old>current else old
        # matches = bf.match(self.old_features,pointcloud[2])
        # matches = sorted(matches, key = lambda x:x.distance)
        # print(np.array(matches).shape)
        matches = matcher.knnMatch(self.old_features[:num],pointcloud[2][:num], k=2) # knnMatch is crucial
        good = []
        for (m1, m2) in matches: # for every descriptor, take closest two matches
            if m1.distance < 0.7 * m2.distance: # best match has to be this much closer than second best
                good.append(m1)
        # print("good/total_matches: ",len(good), len(matches))
        isKeyFrame = len(good)/len(matches) < 0.1
        if isKeyFrame:
            # print("Update new frame as keyframe")
            self.old_features = pointcloud[2]
        self.frames.append(Frame(self.frame_id,isKeyFrame,image,image_coordinates,state,gt_poses,pointcloud))
        self.frame_id += 1
