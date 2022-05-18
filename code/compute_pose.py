import numpy as np
from typing import Union,TypeVar
from feature_point import FeaturePoint
import cv2
from constants import *
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from EssentialMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from DisambiguateCameraPose import DisambiguateCameraPose
from LinearTriangulation import LinearTriangulation
from matplotlib import pyplot as plt

def compute_camera_pose(img_paths, K):
    pose = np.zeros([3,4])
    pointclouds = []
    
    
    for i in range(len(img_paths)):
        img1 = cv2.imread(img_paths[i],cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_paths[i+1],cv2.IMREAD_GRAYSCALE)
               
        points1, points2, points1_desc, points2_desc = getMatches(img1,img2) 


        
        # points1 = np.float32(points1).reshape(-1,1,2)
        # points2 = np.float32(points2).reshape(-1,1,2)

        # F = EstimateFundamentalMatrix(points1, points2)
        F, _ = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC) 
        E = EssentialMatrixFromFundamentalMatrix(F,K)
        R_set,C_set = ExtractCameraPose(E)
        
        

        X_set = []

        for n in range(0, 4):
            X1 = LinearTriangulation(K, np.zeros((3, 1)), np.identity(3), C_set[n].T, R_set[n], np.float32(points1), np.float32(points2))
            X_set.append(X1)

        X, R, C = DisambiguateCameraPose(C_set, R_set, X_set)


        pose[:3,:3]=R
        pose[:,3]=C

        # print(points1.shape)
        # print(img1[np.array([1,2,3,4]),np.array([1,2,3,4])])
        # print(points1[0,0,:].astype(np.int))

        # print(img1[points1[:,0,1].astype(np.int), points1[:,0,0].astype(np.int)].shape)
        for i in range(len(X)):
            pointcloud_color = img1[points1[i,0,1].astype(np.int), points1[i,0,0].astype(np.int)]/255.0
            # print(pointcloud_color.shape)
            # input()
            pointclouds.append(FeaturePoint(position=X[i], color=pointcloud_color, descriptor=points1_desc[i]))
        
        pointcloud_color = img2[points2[:,0,1].astype(np.int), points2[:,0,0].astype(np.int)]/255.0
        pointclouds = [X, pointcloud_color, points2_desc]
        yield img2, points2, pose, pointclouds





def getMatches(img1,img2):

    img1_color = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)

    DETECTOR = SIFT

    kpsA,descA = getKeyPoints(img1,DETECTOR)
    siftImgA = img1_color.copy()
    # cv2.drawKeypoints(siftImgA,kpsA,siftImgA)
    # cv2.imshow("SIFT Keypoints on Image A ",siftImgA)

    
    kpsB,descB = getKeyPoints(img2,DETECTOR)
    siftImgB = img2_color.copy()
    # cv2.drawKeypoints(siftImgB,kpsB,siftImgB)
    # cv2.imshow("SIFT Keypoints on Image B ",siftImgB)

    if DETECTOR==ORB:
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(descA,descB)
        # Sort the matches in asceding order, i.e. closest matches first. 
        matches = sorted(matches, key = lambda x:x.distance)

        matches_img = np.zeros(img1_color.shape)
        matches_img = cv2.drawMatches(img1_color, kpsA, img2_color, kpsB, matches[:50], matches_img, flags=2)
        # cv2.imwrite("Keypoint Matches.png", matches_img)
        # cv2.waitKey(1)
        points1 = np.float32([kpsA[mat.queryIdx].pt for mat in matches]).reshape(-1,1,2)
        points2 = np.float32([kpsB[mat.trainIdx].pt for mat in matches]).reshape(-1,1,2)
        points1_desc = np.array([descA[mat.queryIdx] for mat in matches])
        points2_desc = np.array([descB[mat.trainIdx] for mat in matches])
        # print(points1_desc.shape)
        # matches = bf.match(points1_desc,points1_desc)
        return points1, points2, points1_desc, points2_desc

    elif DETECTOR==SIFT:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descA,descB, k=2)
        
        # Apply ratio test
        good_matches = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good_matches.append(m)

        # cv2.drawMatchesKnn expects list of lists as matches.
        # matches_img = np.zeros(img1_color.shape)
        # matches_img = cv2.drawMatchesKnn(img1_color, kpsA, img2_color, kpsB, good_matches,flags=2)


        points1 = np.float32([kpsA[mat.queryIdx].pt for mat in good_matches]).reshape(-1,1,2)
        points2 = np.float32([kpsB[mat.trainIdx].pt for mat in good_matches]).reshape(-1,1,2)
        points1_desc = np.array([descA[mat.queryIdx] for mat in good_matches])
        points2_desc = np.array([descB[mat.trainIdx] for mat in good_matches])
        # print(points1_desc.shape)
        # matches = bf.match(points1_desc,points1_desc)
        return points1, points2, points1_desc, points2_desc





def getKeyPoints(img,TYPE):
    if TYPE==ORB:
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints with ORB
        kp = orb.detect(img,None)
        # compute the descriptors with ORB
        kp, des = orb.compute(img, kp)
        return kp, des
    
    elif TYPE==SIFT:
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img,None)
        return kp, des

    else:
        raise NotImplementedError