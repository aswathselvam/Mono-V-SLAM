import numpy as np

def ExtractCameraPose(E):

    U, S, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R = []
    C = []
    R.append(np.dot(np.dot(U, W), V_T))
    R.append(np.dot(np.dot(U, W), V_T))
    R.append(np.dot(np.dot(U,W.T), V_T))
    R.append(np.dot(np.dot(U,W.T), V_T))
    C.append(U[:, 2])
    C.append(-U[:, 2])
    C.append(U[:, 2])
    C.append(-U[:, 2])

    for i in range(4):
        if (np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            C[i] = -C[i]

    return R, C



