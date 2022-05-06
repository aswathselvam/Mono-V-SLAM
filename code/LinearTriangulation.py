import numpy as np
import cv2

def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, x[0]], [x[1], x[0], 0]])


def LinearTriangulation(K, C1, R1, C2, R2, x1, x2):

    I = np.identity(3)
    sz = x1.shape[0]
    C1 = C1.reshape(3, 1)
    C2 = C2.reshape(3, 1)
    P1 = K @ np.hstack((R1, C1))
    P2 = K @ np.hstack((R2, C2))

    X1 = np.hstack((x1, np.ones((sz, 1))))
    X2 = np.hstack((x2, np.ones((sz, 1))))

    X = np.zeros((sz, 3))

    for i in range(sz):
        A = np.zeros((4,4))
        A[0,:] = X1[i,0]*P1[2,:] - P1[0,:]
        A[1,:] = X1[i,1]*P1[2,:] - P1[1,:]
        A[2,:] = X2[i,0]*P2[2,:] - P2[0,:]
        A[3,:] = X2[i,1]*P2[2,:] - P2[1,:]
        _, _, Vt = np.linalg.svd(A)
        x = Vt[3,:]
        x = x.reshape(4,)
        x /= x[3]
        X[i, :] = x[0:3].reshape(1,3)

    return X

# def LinearTriangulation(K, c1, r1, c2, r2, p1, p2):
#     c1 = c1.reshape(3, 1)
#     c2 = c2.reshape(3,1)
#
#     r1_c1 = -np.dot(r1, c1)
#     r2_c2 = -np.dot(r2, c2)
#
#     j1 = np.concatenate((r1, r1_c1), axis=1)
#     j2 = np.concatenate((r2, r2_c2), axis=1)
#
#     P1 = np.dot(K, j1)
#     P2 = np.dot(K, j2)
#
#     l_sol = []
#
#     for i in range(len(p1)):
#
#         x_1 = np.array(p1[i])
#         x_2 = np.array(p2[i])
#
#         x_1 = np.reshape(x_1, (2, 1))
#         q = np.array([1])
#         q = np.reshape(q, (1, 1))
#
#         x_1 = np.concatenate((x_1, q), axis=0)
#         x_2 = np.reshape(x_2, (2, 1))
#         x_2 = np.concatenate((x_2, q), axis=0)
#
#         x_1_skew = np.array([[0, -x_1[2][0], x_1[1][0]], [x_1[2][0], 0, -x_1[0][0]], [-x_1[1][0], x_1[0][0], 0]])
#         x_2_skew = np.array([[0, -x_2[2][0], x_2[1][0]], [x_2[2][0], 0, -x_2[0][0]], [-x_2[1][0], x_2[0][0], 0]])
#
#         A1 = np.dot(x_1_skew, P1)
#         A2 = np.dot(x_2_skew, P2)
#         # calculating A and solving using SVD
#         A = np.zeros((6, 4))
#         for i in range(6):
#             if i <= 2:
#                 A[i, :] = A1[i, :]
#             else:
#                 A[i, :] = A2[i - 3, :]
#
#         U, sigma, VT = np.linalg.svd(A)
#         VT = VT[3]
#         VT = VT / VT[-1]
#         l_sol.append(VT)
#
#     l_sol = np.array(l_sol)
#     X = l_sol[:,0:3]
#
#     return X

#
# def LinearTriangulation(K,t1, R1, t2, R2, img1pts, img2pts):
#     t1=t1.reshape(-1,1)
#     t2 = t2.reshape(-1, 1)
#     img1ptsHom = cv2.convertPointsToHomogeneous(img1pts)[:, 0, :]
#     img2ptsHom = cv2.convertPointsToHomogeneous(img2pts)[:, 0, :]
#
#     img1ptsNorm = (np.linalg.inv(K).dot(img1ptsHom.T)).T
#     img2ptsNorm = (np.linalg.inv(K).dot(img2ptsHom.T)).T
#
#     img1ptsNorm = cv2.convertPointsFromHomogeneous(img1ptsNorm)[:, 0, :]
#     img2ptsNorm = cv2.convertPointsFromHomogeneous(img2ptsNorm)[:, 0, :]
#
#     pts4d = cv2.triangulatePoints(np.hstack((R1, t1)), np.hstack((R2, t2)), img1ptsNorm.T, img2ptsNorm.T)
#     pts3d = cv2.convertPointsFromHomogeneous(pts4d.T)[:, 0, :]
#
#     return pts3d

