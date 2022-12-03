"""This file implement the two-phase singular value decomposition algorithm.
"""

import numpy as np
import scipy.linalg
from .HouseHolder import HouseHolder
from .QR import *


def svd_phaseI(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function implement the Golub-Kahan bidiagonalization to reduce the matrix A to bidiagonal form.

    Args:
        A (np.ndarray): The matrix A whose SVD is of our interest.

    Returns:
        B (np.ndarray): The bidiagonalized version of A, with dimension min(m,n) x min(m, n).
        Qt (np.ndarray), and P (np.ndarray): The two transformation matrices such that B = Qt @ A @ P
    """
    B = np.zeros(A.shape)
    m, n = A.shape
    flag_T = False
    if m < n:
        flag_T = True
        A = A.T
        m, n = A.shape

    i = 0
    Qt = HouseHolder(A[:, 0])
    A = Qt @ A

    P = np.zeros((n, n))
    P[0, 0] = 1
    P[1:, 1:] = HouseHolder(A[0, 1:].T)
    A[:, 1:] = A[:, 1:] @ P[1:, 1:]

    for i in range(1, n-1):
        Qit = HouseHolder(A[i:, i])
        A[i:, i:] = Qit @ A[i:, i:]

        Pi = HouseHolder(A[i, (i+1):].T)
        A[i:, i+1:] = A[i:, i+1:] @ Pi

        Qt[i:, :] = Qit @ Qt[i:, :]
        P[:, i+1:] = P[:, i+1:] @ Pi

    i = n-1
    Qit = HouseHolder(A[i:, i])
    A[i:, i:] = Qit @ A[i:, i:]
    Qt[i:, :] = Qit @ Qt[i:, :]

    if flag_T:
        B = A.T
        return B, P.T, Qt.T
    else:
        B = A
        return B, Qt, P


def svd_phaseIIA(B: np.ndarray, Qt: np.ndarray, P: np.ndarray, eigenTest=False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Discard the zero rows or columns
    m, n = B.shape
    if (m > n):
        B = B[:n, :]
        Qt = Qt[:n, :]
    elif (m < n):
        B = B[:, :m]
        P = P[:, :m]

    if eigenTest:
        T, S = scipy.linalg.eigh(B.T@B)
        T, G = scipy.linalg.eigh(B@B.T)
    else:
        T, S = eigh_by_QR(B.T@B)
        T, G = eigh_by_QR(B@B.T)

    U = Qt.T @ G
    Vt = S.T @ P.T
    return U, T**.5, Vt


def svd(A: np.ndarray, phaseII=svd_phaseIIA, eigenTest=False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    B, Q, P = svd_phaseI(A)
    return phaseII(B, Q, P, eigenTest=eigenTest)


if __name__ == "__main__":
    A = np.array([[1, 0, 1],
                  [2, 5**.5, 0],
                  [0, 0, 1],
                  [0, 0, 1]])
    B, Qt, P = svd_phaseI(A)
    print(B)
