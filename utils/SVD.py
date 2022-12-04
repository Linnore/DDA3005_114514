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

    # Naive way:
    # B = A.copy()
    # Qt = np.identity(m)
    # P = np.identity(n)

    # r = min(m, n)
    # for i in range(r-1):
    #     Qit = np.identity(m)
    #     Qit[i:, i:] = HouseHolder(B[i:, i])
    #     B = Qit @ B

    #     Pi = np.identity(n)
    #     Pi[i+1:, i+1: ] = HouseHolder(B[i, i+1:])
    #     B = B @ Pi

    #     Qt = Qit @ Qt
    #     P = P @ Pi
        
    # if m>n:
    #     i = n-1
    #     Qit = np.identity(m)
    #     Qit[i:, i:] = HouseHolder(B[i:, i])
    #     B = Qit @ B
    #     Qt = Qit @ Qt


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

        Pi = HouseHolder(B[i, (i+1):])
        B[i:, i+1:] = B[i:, i+1:] @ Pi

        Qt[i:, :] = Qit @ Qt[i:, :]
        P[:, i+1:] = P[:, i+1:] @ Pi

    if m > n:
        i = r-1
        Qit = HouseHolder(B[i:, i])
        B[i:, i:] = Qit @ B[i:, i:]
        Qt[i:, :] = Qit @ Qt[i:, :]

    return B, Qt, P


def svd_phaseIIA(B: np.ndarray, Qt: np.ndarray, P: np.ndarray, eigen=eigh_by_QR, tol=1e-15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function implement the phaseII of SVD following the proposed procedure A in project description.

    Args:
        B (np.ndarray): The bidiagonalized version of A such that A = Qt @ B @ P
        Qt (np.ndarray): An orthogonal matrix that performs the left Householder transformation on A.
        P (np.ndarray): An orthogonal matrix that performs the rigth Householder transformation on A.
        eigen (function, optional): The eigen solver used to get eigenvalues and eigen vectors of B @B.T or B.T @ B. 
            Defaults to the one we implement: eigh_by_QR. Possible candidates could be scipy.linalg.eigh,
            sicpy.linalg.eigh_tridiagonal.
        tol (float): The tolerance for judging zero.

    Returns:
        U (np.ndarray): The matrix containing left singular vectors of A as columns 
        S (np.ndarray): An 1d array containing the singular values in descending order.
        Vt (np.ndarray): The transepose of the matrix containing right singular vectors of A as columns.
    """

    # Discard the zero rows or columns
    m, n = B.shape
    if m < n:
        B = B[:, :m]
        P = P[:, :m]
        n = m
    elif m > n:
        B = B[:n]
        Qt = Qt[:n]
        m = n

    # Eigen decomposition of B'@B = S @ T @ S'
    # Eigen decomposition of B@B' = G @ T @ G'
    T, G = eigen(B@B.T)
    _, S = eigen(B.T@B)
    # Todo: Save one eigen decomposition
    # How to apple rowwise backward and forward substitution on singular B?

    U = Qt.T @ G
    Vt = S.T @ P.T
    # T[np.abs(T) < tol] = 0
    return U, T**.5, Vt


def svd(A: np.ndarray, phaseII=svd_phaseIIA, eigen=eigh_by_QR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    B, Qt, P = svd_phaseI(A)
    return phaseII(B, Qt, P, eigen=eigen)


if __name__ == "__main__":
    A = np.array([[1, 0, 1],
                  [2, 5**.5, 0],
                  [0, 0, 1],
                  [0, 0, 1]])
    B, Qt, P = svd_phaseI(A)
    print(B)
