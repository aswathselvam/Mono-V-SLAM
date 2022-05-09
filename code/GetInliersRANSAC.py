import numpy as np
import cv2
from FundamentalMatrix import eight_point_F

def normalise(pts1, pts2, T):
  ones = np.ones((1, pts1.shape[0]))
  pts1_norm = np.vstack((pts1.T, ones))
  pts2_norm = np.vstack((pts2.T, ones))

  pts1_norm = (T @ pts1_norm).T[:, 0:2]
  pts2_norm = (T @ pts2_norm).T[:, 0:2]
  return pts1_norm, pts2_norm

def GetInliersRANSAC2(points1, points2, indices,w,h):
    F, mask = cv2.findFundamentalMat(points1.reshape(-1,1,2),points2.reshape(-1,1,2),cv2.FM_RANSAC)
    mask = mask.ravel()
    in_points_x1 = points1[mask == 1]
    in_points_x2 = points2[mask == 1]
    inlier_index = (np.argwhere(mask)).reshape(-1)

    return F, in_points_x1, in_points_x2, inlier_index

def GetInliersRANSAC(points1, points2, w, h):
    T = np.array([[2 / w, 0, -1], [0, 2 / h, -1], [0, 0, 1]])
    points1_norm, points2_norm = normalise(points1, points2, T)
   
    N = 1000

    thresh = 0.01
    inliers_max = 0
    F_final = []

    k=0

    while k < N:

        idx = np.random.randint(0, points1.shape[0], size=8)

        # F = EstimateFundamentalMatrix(points1_norm[idx, :], points2_norm[idx, :])
        F = eight_point_F(points1_norm[idx, :], points2_norm[idx, :])

        in_a = []
        in_b = []
        ind = []
        inlier_count = 0
        for i in range(points1.shape[0]):
            matches_aa = np.append(points1_norm[i, :], 1)
            matches_bb = np.append(points2_norm[i, :], 1)
            error = np.dot(matches_aa, F.T)
            error = np.dot(error, matches_bb.T)
            if abs(error) < thresh:
                in_a.append(points1[i, :])
                in_b.append(points2[i, :])
                inlier_count += 1

        if inliers_max < inlier_count:

            inliers_max = inlier_count

            in_points_x1 = in_a
            in_points_x2 = in_b

            F_final = F

        k += 1

    in_points_x1 = np.array(in_points_x1)
    in_points_x2 = np.array(in_points_x2)

    F = T.T @ F_final @ T
    U,s,Vt = np.linalg.svd(F)
    s[-1] = 0.0
    F = U @ np.diag(s) @ Vt
    F /=F[2,2]

    return F, in_points_x1, in_points_x2
