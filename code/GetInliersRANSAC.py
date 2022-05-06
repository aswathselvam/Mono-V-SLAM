import numpy as np
import cv2
from scripts.EstimateFundamentalMatrix import EstimateFundamentalMatrix, eight_point_F

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

def GetInliersRANSAC(points1, points2, indices,w,h):
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
                ind.append(indices[i])
                inlier_count += 1

        if inliers_max < inlier_count:

            inliers_max = inlier_count

            in_points_x1 = in_a
            in_points_x2 = in_b
            inlier_index = ind

            F_final = F

        k += 1

    in_points_x1 = np.array(in_points_x1)
    in_points_x2 = np.array(in_points_x2)
    inlier_index = np.array(inlier_index)

    F = T.T @ F_final @ T
    U,s,Vt = np.linalg.svd(F)
    s[-1] = 0.0
    F = U @ np.diag(s) @ Vt
    F /=F[2,2]

    return F, in_points_x1, in_points_x2, inlier_index

def inlier_filter(Mx, My, M, n_images,w,h):
    print("Finding Inliers")

    outlier_indices = np.zeros(M.shape)
    for i in range(1, n_images):
        for j in range(i + 1, n_images + 1):
            img1 = i
            img2 = j

            output = np.logical_and(M[:, img1 - 1], M[:, img2 - 1])
            indices, = np.where(output == True)
            if (len(indices) < 8):
                continue
            pts1 = np.hstack((Mx[indices, img1 - 1].reshape((-1, 1)), My[indices, img1 - 1].reshape((-1, 1))))
            pts2 = np.hstack((Mx[indices, img2 - 1].reshape((-1, 1)), My[indices, img2 - 1].reshape((-1, 1))))

            _, inliers_a, inliers_b, inlier_index = GetInliersRANSAC2(np.float32(pts1), np.float32(pts2), indices, w, h)

            for k in indices:
                if (k not in inlier_index):
                    M[k, i - 1] = 0
                    outlier_indices[k, i - 1] = 1
                    outlier_indices[k, j - 1] = 1
            print('Image ',i,' and ',j,' done')
    return M, outlier_indices
