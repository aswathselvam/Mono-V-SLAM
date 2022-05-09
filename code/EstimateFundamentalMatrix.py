import numpy as np
from random import sample

def EstimateFundamentalMatrix(points1, points2):
    A=[]

    for i in range(0,points1.shape[0]):
        x1, y1 = points1[i][0], points1[i][1]
        x_1, y_1 = points2[i][0], points2[i][1]
        A.append([x1*x_1, x1*y_1, x1, y1*x_1, y1*y_1, y1, x_1, y_1, 1])
    Am = np.array(A)

    U, S, V = np.linalg.svd(Am)
    V = V.T

    f_val = V[:, -1]
    f_mat = f_val.reshape((3, 3))

    Uf, Sf, Vf = np.linalg.svd(f_mat)
    Sf[-1] = 0

    sigma_final = np.zeros(shape=(3, 3))
    sigma_final[0][0] = Sf[0]
    sigma_final[1][1] = Sf[1]
    sigma_final[2][2] = Sf[2]

    F = np.dot(Uf, sigma_final)
    F = np.dot(F, Vf)
    F = F / F[2, 2]

    return F


def normalise(pts1, pts2, T):
    ones = np.ones((1, pts1.shape[0]))
    # converting into homogenous form for
    # transforming/normalising
    pts1_norm = np.vstack((pts1.T, ones))
    pts2_norm = np.vstack((pts2.T, ones))

    pts1_norm = (T @ pts1_norm).T[:, 0:2]
    pts2_norm = (T @ pts2_norm).T[:, 0:2]
    return pts1_norm, pts2_norm


def eight_point_F(p1, p2):
    """
    Takes in 8 coressponding points and returns F matrix
    """
    n = p1.shape[0]
    a = []
    for i in range(0, n):
        a.append(
            [p1[i][0] * p2[i][0], p1[i][0] * p2[i][1], p1[i][0], p1[i][1] * p2[i][0], p1[i][1] * p2[i][1], p1[i][1],
             p2[i][0], p2[i][1], 1])
    A = np.array(a)
    _, _, Vh = np.linalg.svd(A)
    x = Vh[-1, :]
    F = np.array([[x[0], x[3], x[6]],
                  [x[1], x[4], x[7]],
                  [x[2], x[5], x[8]]])
    U, S, Vh = np.linalg.svd(F)
    S[-1] = 0.0
    F = U @ np.diag(S) @ Vh
    F /= F[2, 2]

    return F


def get_eight_random_points(pts1, pts2):
    n = pts1.shape[0]
    nums = sample(range(0, n), 8)
    p1 = pts1[nums]
    p2 = pts2[nums]

    return p1, p2


def ransac_F(pts1, pts2):
    n = 0
    Fs = None
    matchesMask = None
    eps = 0.01
    iter = 1000
    ones = np.ones((pts1.shape[0], 1))
    X1 = np.hstack((pts1, ones))
    X2 = np.hstack((pts2, ones))

    for i in range(0, iter):
        p1, p2 = get_eight_random_points(pts1, pts2)
        F = eight_point_F(p1, p2)
        val = np.absolute(np.sum(X2.T * (F @ X1.T), axis=0))
        inliners_mask = val < eps
        num_inliners = (np.argwhere(inliners_mask)).shape[0]
        if n < num_inliners:
            n = num_inliners
            Fs = F
            matchesMask = inliners_mask
    print(val.shape)
    print("no. of inliers : ", n)
    return Fs, matchesMask


def get_F(pts1, pts2, w,h):
    T = np.array([[2 / w, 0, -1], [0, 2 / h, -1], [0, 0, 1]])
    pts1_norm, pts2_norm = normalise(pts1, pts2, T)
    Fhat, matchesMask = ransac_F(pts1_norm, pts2_norm)
    F = T.T @ Fhat @ T
    U, s, Vt = np.linalg.svd(F)
    s[-1] = 0.0
    F = U @ np.diag(s) @ Vt
    F /= F[2, 2]
    return F, matchesMask