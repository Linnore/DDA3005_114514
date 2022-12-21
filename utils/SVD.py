"""This file implement the two-phase singular value decomposition algorithm.
"""

import numpy as np
from .HouseHolder import HouseHolder, HouseHolder_update
from .QR import *
from time import time
import sys
sys.setrecursionlimit(4500)


def svd_phaseI(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function implement the Golub-Kahan bidiagonalization to reduce the matrix A to bidiagonal form.
    Note that this input matrix must be a tall matrix, i.e. m>=n.

    Args:
        A (np.ndarray): The matrix A whose SVD is of our interest.

    Returns:
        B (np.ndarray): The bidiagonalized version of A, with dimension n x n.
        Qt (np.ndarray), and P (np.ndarray): The two transformation matrices such that B = Qt @ A @ P,
            where Qt is m by m, P is n by n.
    """
    m, n = A.shape
    B = A.copy()

    Qt = np.identity(m)
    P = np.identity(n)
    r = min(m, n)

    for i in range(r-1):
        w, alpha = HouseHolder(B[i:, i])
        w = w.reshape(-1,1)
        HouseHolder_update(B[i:, i:], w, alpha)
        Qt[i:, :] = Qt[i:, :] - w @ w.T @ Qt[i:, :]

        w, alpha = HouseHolder(B[i, i+1:])
        w = w.reshape(-1,1)
        HouseHolder_update(B[i:, i+1:].T, w, alpha)
        P[:, i+1:] = P[:, i+1:] - P[:, i+1:] @ w @ w.T

    i = r-1
    v, alpha = HouseHolder(B[i:, i])
    if alpha == 0:
        w = v.reshape((-1, 1))
    else:
        w = (np.sqrt(2)/norm(v) * v).reshape((-1, 1))
    HouseHolder_update(B[i:, i:], w, alpha)
    Qt[i:, :] = Qt[i:, :] - w @ (w.T @ Qt[i:, :])

    return B, Qt, P
    # Todo, form Q can be even faster!


def fastMult_upper_bidiagonal(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """This function exploit the struction of a bidiagonal matrix to compute A@B in O(n^2), where A
    is a general matrix and B is the upper bidiagonal matrix.

    Args:
        A (np.ndarray): _description_
        B (np.ndarray): _description_

    Returns:
        (np.ndarray): The product A@B
    """
    m, n = (A.shape[0], B.shape[-1])
    result = np.zeros((m, n))
    result[:, 0] = B[0, 0]*A[:, 0]
    for i in range(1, n):
        result[:, i] = B[i-1, i]*A[:, i-1] + B[i, i]*A[:, i]
    return result


def fastMult_lower_bidiagonal(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """This function exploit the struction of a bidiagonal matrix to compute A@B in O(n^2), where A
    is a general matrix and B is the lower bidiagonal matrix.

    Args:
        A (np.ndarray): _description_
        B (np.ndarray): _description_

    Returns:
        (np.ndarray): The product A@B
    """
    m, n = (A.shape[0], B.shape[-1])
    result = np.zeros((m, n))
    result[:, -1] = B[-1, -1]*A[:, -1]
    for i in range(n-2, -1, -1):
        result[:, i] = B[i+1, i]*A[:, i+1] + B[i, i]*A[:, i]
    return result


def svd_phaseII(B: np.ndarray, Qt: np.ndarray, P: np.ndarray, phaseII: str, eigen=eigh_by_QR, tol=1e-13) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function implement the phaseII of SVD following the proposed procedure A in project description.

    Args:
        B (np.ndarray): The bidiagonalized version of A such that A = Qt @ B @ P. Note that B is upper triangular.
        Qt (np.ndarray): An orthogonal matrix that performs the left Householder transformation on A.
        P (np.ndarray): An orthogonal matrix that performs the rigth Householder transformation on A.
        phaseII (str): It indicates which phaseII is calling. Possible values are "A", "B", "B1", "B2".
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
    B = B[:n]
    Qt = Qt[:n]
    m = n
    # Eigen decomposition of B@B' = G @ T @ G'
    # B = GTS'; G'B = TS'
    if phaseII == "A":
        T, G = eigen(fastMult_lower_bidiagonal(B, B.T))
    else:
        T, G = eigen(B)
    nonzero_idx = np.abs(T) > tol
    T = T[nonzero_idx]
    G = G[:, nonzero_idx]
    sigma = T**.5
    S = (fastMult_upper_bidiagonal(G.T, B)).T/sigma

    U = Qt.T @ G
    Vt = S.T @ P.T
    return U, sigma, Vt


def svd(A: np.ndarray, phaseII='Default', eigen=eigh_by_QR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    Args:
        A (np.ndarray): _description_
        phaseII (str, optional): Possible choices: 'A', 'B1' ,and 'B2'. Defaults to 'Default'.
        eigen (_type_, optional): _description_. Defaults to eigh_by_QR.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: _description_
    """
    m, n = A.shape
    flipped = False
    if m < n:
        flipped = True
        A = A.T
        m, n = A.shape

    p1_begin = time()
    B, Qt, P = svd_phaseI(A)
    p1_end = time()
    print("phaseI: {:4f}s".format(p1_end - p1_begin))

    if phaseII == 'A':
        eigenSolver = eigh_by_QR
    elif phaseII == 'B1' or phaseII == 'B':
        eigenSolver = eigh_of_BBT
    elif phaseII == 'B2':
        eigenSolver = eigh_of_BBT_optional
    else:
        phaseII = "A"
        eigenSolver = eigen

    p2_begin = time()
    U, S, Vt = svd_phaseII(B, Qt, P, phaseII, eigen=eigenSolver)
    p2_end = time()
    print("phaseII: {:4f}s".format(p2_end - p2_begin))

    if flipped:
        return Vt.T, S, U.T
    else:
        return U, S, Vt


# def svd_phaseI(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """This function implement the Golub-Kahan bidiagonalization to reduce the matrix A to bidiagonal form.

#     Args:
#         A (np.ndarray): The matrix A whose SVD is of our interest.

#     Returns:
#         B (np.ndarray): The bidiagonalized version of A, with dimension min(m,n) x min(m, n).
#         Qt (np.ndarray), and P (np.ndarray): The two transformation matrices such that B = Qt @ A @ P
#     """
#     m, n = A.shape
#     B = A.copy()
#     i = 0
#     Qt = HouseHolder(B[:, 0])
#     B = Qt @ B

#     r = min(m, n)
#     P = np.zeros((n, n))
#     P[0, 0] = 1
#     P[1:, 1:] = HouseHolder(B[0, 1:].T)
#     B[:, 1:] = B[:, 1:] @ P[1:, 1:]

#     for i in range(1, r-1):
#         Qit = HouseHolder(B[i:, i])
#         # print(np.sum(np.abs(Qit-Qit.T) < 1e-14)==B[i:, i].shape[0]**2)
#         B[i:, i:] = Qit @ B[i:, i:]

#         Pi = HouseHolder(B[i, (i+1):])
#         B[i:, i+1:] = B[i:, i+1:] @ Pi

#         Qt[i:, :] = Qit @ Qt[i:, :]
#         P[:, i+1:] = P[:, i+1:] @ Pi

#     i = r-1
#     Qit = HouseHolder(B[i:, i])
#     B[i:, i:] = Qit @ B[i:, i:]
#     Qt[i:, :] = Qit @ Qt[i:, :]

#     return B, Qt, P
