import numpy as np
import sys


def EssentialMatrixFromFundamentalMatrix(F, K):

    E = np.dot(K.T, np.dot(F, K))
    U, S, V_T = np.linalg.svd(E)

    E = np.dot(U, np.dot(np.diag([1, 1, 0]), V_T))
    return E
