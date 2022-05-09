import numpy as np
import scipy.optimize as opt


def NonLinearTriangulation(K, x1, x2, X_init, R1, C1, R2, C2):

    sz = x1.shape[0]

    init = X_init.flatten()

    optimized_params = opt.least_squares(fun=minimizeFunction, x0=init, method="dogbox", args=[K, x1, x2, R1, C1, R2, C2])

    X = np.reshape(optimized_params.x, (sz, 3))

    return X


def minimizeFunction(init, K, x1, x2, R1, C1, R2, C2):

    sz = x1.shape[0]
    X = np.reshape(init, (sz, 3))

    I = np.identity(3)
    C2 = np.reshape(C2, (3, -1))

    X = np.hstack((X, np.ones((sz, 1))))

    P1 = K @ np.hstack((R1, C1)) #np.dot(K, np.dot(R1, np.hstack((I, -C1))))
    P2 = K @ np.hstack((R2, C2)) #np.dot(K, np.dot(R2, np.hstack((I, -C2))))
    #print(P1, P2)


    u1 = np.divide((np.dot(P1[0, :], X.T).T), (np.dot(P1[2, :], X.T).T))
    v1 = np.divide((np.dot(P1[1, :], X.T).T), (np.dot(P1[2, :], X.T).T))
    u2 = np.divide((np.dot(P2[0, :], X.T).T), (np.dot(P2[2, :], X.T).T))
    v2 = np.divide((np.dot(P2[1, :], X.T).T), (np.dot(P2[2, :], X.T).T))

    error1 = ((x1[:, 0] - u1) + (x1[:, 1] - v1))
    error2 = ((x2[:, 0] - u2) + (x2[:, 1] - v2))
    #     print(error1.shape)
    error = sum(error1, error2)

    return sum(error)