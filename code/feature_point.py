import numpy as np
class FeaturePoint():
    def __init__(self,position, color, descriptor=None):
        self.position = position
        self.descriptor = descriptor
        self.color = color

    def __add__(self, other):
        return FeaturePoint(np.sum([self.position,other], axis=0), self.color, self.descriptor)
