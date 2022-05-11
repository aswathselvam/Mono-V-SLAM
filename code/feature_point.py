import numpy as np
class FeaturePoint():
    def __init__(self,position, color, descriptor=None):
        self.position = position
        self.descriptor = descriptor
        self.color = color
        self.R=np.eye(3)
        self.T=np.zeros(3)

    def __add__(self, other):
        return FeaturePoint(np.sum([self.position,other], axis=0), self.color, self.descriptor)

    def update(self, pose):
        self.R = self.R@pose[:3,:3]
        #self.T = self.T + self.R@pose[:3,3]
        # self.T = self.position + self.R@pose[:3,3]
        # print(np.linalg.det(self.T))
        self.position = self.position + pose[:3,3]
        # self.position = self.T[0]
        # self.position = self.T[1]
        # self.position = self.T[2]
        