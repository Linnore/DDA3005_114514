"""This file implement the two-phase singular value decomposition algorithm.
"""

import numpy as np
import scipy
from .QR_Factorization import HouseHolder, HouseHolder_update
from .EVD import *
from .Bidiagonal_fastMult import *
from time import time

def svd_phaseI(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function implement the Golub-Kahan bidiagonalization to reduce the matrix A to bidiagonal form B,
    where A = Qt @ B @ P.
        Note that this input matrix must be a tall matrix, i.e. m>=n. 
        We achieve O(n^3) instead of O(n^4) when forming the orthogonal factors Qt and P. The procedure is very much
    like performing HouseHolder QR factorization left and right at the same time.

    Args:
        A (np.ndarray): The matrix A whose SVD is of our interest.

    Returns:
        B (np.ndarray): The bidiagonalized version of A, with dimension n x n.
        Qt and P (np.ndarray): The two transformation matrices such that B = Qt @ A @ P,
            where Qt is m by m, P is n by n.
    """
    m, n = A.shape
    if m < n:
        print("PhaseI only support tall matrix (m>=n) input!")
        raise()
    B = A.copy()

    Q = np.identity(m)
    wq = np.empty((m, m))
    P = np.identity(n)
    wp = np.empty((n-1, n-1))

    for i in range(n-1):
        wq[i, i:], alpha = HouseHolder(B[i:, i])
        HouseHolder_update(B[i:, i:], wq[i, i:], alpha)

        wp[i, i:], alpha = HouseHolder(B[i, i+1:])
        HouseHolder_update(B[i:, i+1:].T, wp[i, i:], alpha)
    i = n-1
    wq[i, i:], alpha = HouseHolder(B[i:, i])
    HouseHolder_update(B[i:, i:], wq[i, i:], alpha)

    # Forming Q and P reversely:
    for i in range(n-1, 0, -1):
        w = wq[i, i:].reshape(-1, 1)
        Q[i:, i:] = Q[i:, i:] - w @ (w.T @ Q[i:, i:])

        w = wp[i-1, i-1:].reshape(-1, 1)
        P[i:, i:] = P[i:, i:] - w @ (w.T @ P[i:, i:])
    i = 0
    w = wq[i, i:].reshape(-1, 1)
    Q[i:, i:] = Q[i:, i:] - w @ (w.T @ Q[i:, i:])

    return B, Q.T, P


def svd_phaseII(B: np.ndarray, Qt: np.ndarray, P: np.ndarray, phaseII: str, eigen=eigh_by_QR, less_as_zero=1e-15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function implement the phaseII of SVD following the proposed procedure A in project description.

    Args:
        B (np.ndarray): The bidiagonalized version of A such that A = Qt @ B @ P. Note that B is upper triangular.
        Qt (np.ndarray): An orthogonal matrix that performs the left Householder transformation on A.
        P (np.ndarray): An orthogonal matrix that performs the rigth Householder transformation on A.
        phaseII (str): It indicates which phaseII is calling. Possible values are "A", "B", "B1", "B2".
        eigen (function, optional): The eigen solver used to get eigenvalues and eigen vectors of B @B.T or B.T @ B. 
            Defaults to the one we implement: eigh_by_QR. Possible candidates could be scipy.linalg.eigh,
            sicpy.linalg.eigh_tridiagonal.
        less_as_zero (float): The tolerance for judging zero.

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
    B_nonzero_idx = np.abs(B.diagonal()) > less_as_zero
    B = B[B_nonzero_idx]
    B = B[:, B_nonzero_idx]
    Qt = Qt[B_nonzero_idx]
    P = P[:, B_nonzero_idx]
    # B = Qt @ A @ P, A = Q @ B @ P'
    # Eigen decomposition of B'@B = G @ T @ G' = G @ sigma @ sigma @ G'
    # B = G @ sigma @ S'; G'@ B = sigma @ S'
    if phaseII == "A":
        T, G = eigen(fastMult_upper_bidiagonal(B.T, B))
        sigma = T**.5
    else:
        sigma, G = eigen(B, return_singularV_of_B=True)
        S = (fastMult_upper_bidiagonal(B, G))/sigma
        U = Qt.T @ S.T
        Vt = G @ P.T
        
    S = (fastMult_upper_bidiagonal(B, G))/sigma
    Vt = G.T @ P.T
    U = Qt.T @ B @ P.T @ Vt.T / sigma
    return U, sigma, Vt


def svd(A: np.ndarray, phaseII='Default', eigen=eigh_by_QR, timed=False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function compute the economic SVD of a real matrix A.

    Args:
        A (np.ndarray): A real matrix.
        phaseII (str, optional): Possible choices: 'A', 'B1', 'B2', and 'C'. Defaults to 'Default'.
        eigen (_type_, optional): _description_. Defaults to eigh_by_QR.

    Returns:
        U, S, Vt (np.ndarray): The economic SVD of matrix A s.t. A=U@diag(S)@Vt.
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
    print("phaseI: {:.4f}s".format(p1_end - p1_begin))

    if phaseII == 'A' or phaseII == 'A1':
        phaseII = 'A'
        eigenSolver = eigh_by_QR
    elif phaseII == 'A2':
        phaseII = 'A'
        eigenSolver = eigh_by_QR_optional
    elif phaseII == 'B' or phaseII == 'B1':
        eigenSolver = eigh_of_BTB
    elif phaseII == 'B2':
        eigenSolver = eigh_of_BTB_optional
    elif phaseII == 'C':
        eigenSolver = eigh_of_BTB_cheat
    else:
        phaseII = "A"
        eigenSolver = eigen

    p2_begin = time()
    U, S, Vt = svd_phaseII(B, Qt, P, phaseII, eigen=eigenSolver)
    p2_end = time()
    print("phaseII: {:.4f}s".format(p2_end - p2_begin))

    if flipped:
        if timed:
            return Vt.T, S, U.T, p1_end - p1_begin, p2_end - p2_begin
        else:
            return Vt.T, S, U.T
    else:
        if timed:
            return U, S, Vt, p1_end - p1_begin, p2_end - p2_begin
        else:
            return U, S, Vt

def accuracy_test(A, U, S, Vt, acc=1e-8): 
    _, ref_sv, _ = scipy.linalg.svd(A)   
    m, n = A.shape
    print("Percentage of entrices successfully recovered by SVD with accuracy: {}".format(acc))
    print(np.sum(np.abs(U@np.diag(S)@Vt - A)< acc) / (n*m) * 100, "%")

    print("Percentage of singular values with accuracy: {}".format(acc))
    print(np.sum(np.abs(S - ref_sv[:S.size])< acc) / S.size * 100, "%")

    print("Max error of singular values:")
    print(np.abs(ref_sv[:S.size] - S).max())
    
    return (np.sum(np.abs(U@np.diag(S)@Vt - A)< acc) / (n*m) * 100, \
            np.sum(np.abs(S - ref_sv[:S.size])< acc) / S.size * 100, \
            np.abs(ref_sv[:S.size] - S).max())

def is_orthogonal(A, tol=1e-4, silence=True):
    m = A.shape[0]
    Q = A @ A.T
    if not silence:
        print(np.linalg.norm(np.identity(m) - Q))
    return np.linalg.norm(np.identity(m) - Q) < tol
