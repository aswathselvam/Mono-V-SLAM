import numpy as np
from typing import Union,TypeVar
from feature_point import FeaturePoint
import cv2
from constants import *
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose


def compute_camera_pose(img_paths, K):
    pose = np.zeros([3,4])
    pointclouds = FeaturePoint( np.zeros([3,3]))
    
    
    for i in range(len(img_paths)):
        img1 = cv2.imread(img_paths[i],cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_paths[i+1],cv2.IMREAD_GRAYSCALE)
               
        points1,points2 = getMatches(img1,img2) 
        
        F = EstimateFundamentalMatrix(points1, points2)
        E = EssentialMatrixFromFundamentalMatrix(F,K)
        R,T = ExtractCameraPose(E)
        
        
        
        yield pose, pointclouds





def getMatches(img1,img2):

    img1_color = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)


    kpsA,descA = getKeyPoints(img1,ORB)
    siftImgA = img1_color.copy()
    cv2.drawKeypoints(siftImgA,kpsA,siftImgA)
    cv2.imshow("SIFT Keypoints on Image A ",siftImgA)

    
    kpsB,descB = getKeyPoints(img2,ORB)
    siftImgB = img2_color.copy()
    cv2.drawKeypoints(siftImgB,kpsB,siftImgB)
    cv2.imshow("SIFT Keypoints on Image B ",siftImgB)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(descA,descB)

    # Sort the matches in asceding order, i.e. closest matches first. 
    matches = sorted(matches, key = lambda x:x.distance)


    matches_img = np.zeros(img1_color.shape)
    matches_img = cv2.drawMatches(img1_color, kpsA, img2_color, kpsB, matches[:50], matches_img, flags=2)
    cv2.imshow("Keypoint Matches", matches_img)

    points1 = np.float32([kpsA[mat.queryIdx].pt for mat in matches]).reshape(-1,1,2)
    points2 = np.float32([kpsB[mat.trainIdx].pt for mat in matches]).reshape(-1,1,2)

    return points1, points2



def getKeyPoints(img,TYPE):
    if TYPE==ORB:
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints with ORB
        kp = orb.detect(img,None)
        # compute the descriptors with ORB
        kp, des = orb.compute(img, kp)
        return kp, des

    else:
        raise NotImplementedError