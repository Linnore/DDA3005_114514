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
    m, n = A.shape

    B = A.copy()
    i = 0
    Qt = HouseHolder(B[:, 0])
    B = Qt @ B

    r = min(m, n)
    P = np.zeros((n, n))
    P[0, 0] = 1
    P[1:, 1:] = HouseHolder(B[0, 1:].T)
    B[:, 1:] = B[:, 1:] @ P[1:, 1:]

    for i in range(1, r-1):
        Qit = HouseHolder(B[i:, i])
        B[i:, i:] = Qit @ B[i:, i:]

        Pi = HouseHolder(B[i, (i+1):].T)
        B[i:, i+1:] = B[i:, i+1:] @ Pi

        Qt[i:, :] = Qit @ Qt[i:, :]
        P[:, i+1:] = P[:, i+1:] @ Pi

    i = r-1
    Qit = HouseHolder(B[i:, i])
    B[i:, i:] = Qit @ B[i:, i:]
    Qt[i:, :] = Qit @ Qt[i:, :]

    return B, Qt, P


def svd_phaseIIA(B: np.ndarray, Qt: np.ndarray, P: np.ndarray, eigen=eigh_by_QR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function implement the phaseII of SVD following the proposed procedure A in project description.

    Args:
        B (np.ndarray): The bidiagonalized version of A such that A = Qt @ B @ P
        Qt (np.ndarray): An orthogonal matrix that performs the left Householder transformation on A.
        P (np.ndarray): An orthogonal matrix that performs the rigth Householder transformation on A.
        eigen (function, optional): The eigen solver used to get eigenvalues and eigen vectors of B @B.T or B.T @ B. 
            Defaults to the one we implement: eigh_by_QR. Possible candidates could be scipy.linalg.eigh,
            sicpy.linalg.eigh_tridiagonal.

    Returns:
        U (np.ndarray): The matrix containing left singular vectors of A as columns 
        S (np.ndarray): An 1d array containing the singular values in descending order.
        Vt (np.ndarray): The transepose of the matrix containing right singular vectors of A as columns.
    """

    # Discard the zero rows or columns
    m, n = B.shape
    if (m > n):
        B = B[:n, :]
        Qt = Qt[:n, :]
        m = n
    elif (m < n):
        B = B[:, :m]
        P = P[:, :m]
        n = m

    # Eigen decomposition of B'@B = S @ T @ S'
    T, G = eigen(B@B.T)

    # For B@B' = G @ T @ G', solve S by B @ S = G @ T
    # Todo: 
    T, S = eigen(B.T@B)

    U = Qt.T @ G
    Vt = S.T @ P.T
    return U, T**.5, Vt


def svd(A: np.ndarray, phaseII=svd_phaseIIA, eigen=eigh_by_QR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    B, Q, P = svd_phaseI(A)
    return phaseII(B, Q, P, eigen=eigen)


if __name__ == "__main__":
    A = np.array([[1, 0, 1],
                  [2, 5**.5, 0],
                  [0, 0, 1],
                  [0, 0, 1]])
    B, Qt, P = svd_phaseI(A)
    print(B)
