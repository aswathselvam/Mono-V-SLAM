import numpy as np

class State():
    def __init__(self, x=0,y=0,z=0):
        self.x=x
        self.y=y
        self.z=z
        self.R=np.eye(3)
        self.T=np.zeros(3)

    def __add__(self, other):
        return State(self.x+other.x, self.y+other.y, self.z+other.z )

    def update(self, pose):
        self.R = self.R@pose[:3,:3]
        self.T = self.T + self.R@pose[:3,3]
        # self.T = self.T*0.9
        # self.T = self.R@pose[:3,3]
        # print(np.linalg.det(self.T))

        self.x = self.T[0]
        self.y = self.T[1]
        self.z = self.T[2]


    # def __iadd__(self, other):
        # self.x=self.x + other.x
        # self.y=self.y + other.y
        # self.z=self.z + other.z