"""This file implement the two-phase singular value decomposition algorithm.
"""

import numpy as np
from numpy.linalg import norm
import scipy
import scipy.linalg


def hello(test):
    print(test)


def HouseHolder(a: np.ndarray) -> np.ndarray:
    """This function implement the HouseHolder transformation on a given vector 'a'.

    Args:
        a (np.ndarray): The vector of interest. 

    Returns:
        H (np.ndarray): The HouseHolder reflection matrix H with Ha = [alpha, 0, ..., 0], where abs(alpha) is the L2 norm of vector 'a'.
    """
    alpha = np.linalg.norm(a)
    n = a.shape[0]

    # To minuate cancellation effects
    if (a[0] > 0):
        alpha = -alpha

    e1 = np.zeros(n)
    e1[0] = 1
    v = a - alpha*e1
    H = np.identity(n) - 2 * np.outer(v, v) / (v.T @ v)

    return H


def svd_phaseI(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function implement the Golub-Kahan bidiagonalization to reduce the matrix A to bidiagonal form.

    Args:
        A (np.ndarray): The matrix A whose SVD is of our interest.

    Returns:
        B (np.ndarray): The bidiagonalized version of A, with dimension min(m,n) x min(m, n).
        Qt (np.ndarray), and P (np.ndarray): The two transformation matrices such that B = Qt @ A @ P
    """
    B = np.array(A.shape)
    m, n = A.shape
    flag_T = False
    if m < n:
        flag_T = True
        A = A.T

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

    if flag_T:
        B = A.T[:m, :m]
        Qt = Qt[:m, :m]
        P = P[:m, :m]
    else:
        B = A[:n, :n]
        Qt = Qt[:n, :n]
        P = P[:n, :n]
    return B, Qt, P


def Rayleigh_Quotient_Shift(A: np.ndarray) -> int:
    return A[-1, -1]


def Wilkinson_Shift(A: np.ndarray) -> int:
    T = A[-2, -2] + A[-1, -1]
    D = A[-2, -2]*A[-1, -1] - A[-1, -2]*A[-2, -1]
    e1 = T/2 + (T**2/4 - D)**.5
    e2 = T/2 - (T**2/4 - D)**.5
    return e1 if abs(e1-A[-1, -1]) < abs(e2-A[-1, -1]) else e2


def eigh_by_QR(A: np.ndarray, shift=Wilkinson_Shift, qr=scipy.linalg.qr, tol=1e-16, maxn=1000) -> tuple[np.ndarray, np.ndarray]:
    """This function applies the QR algorithm with deflation on the symmetric matrix A 
    to compute its eigenvalue decomposition A = Q@T@Q', where Q contains the eigenvectors
    and T is the diagonal matrix containing the corresponding eigenvalues.

    Args:
        A (np.ndarray): The matrix of interest. Note that it must be real and symmetrix.
        qr (function): An QR decomposition function qr(A) that returns Q and R factors given a matrix A. Defaults to scipy.linalg.qr.
        tol (float, optional): The torlerence for each defletion step. Defaults to 1e-16.
        maxn (int, optional): Maximum iterations at each defletion step. Defaults to 1000.

    Returns:
        T (np.ndarray): An 1d array that contains the eigenvalues of A in descending order.
        Q (np.ndarray): A 2d array (matrix) that contains the corresponding eigenvectors as columns.
    """
    n = A.shape[0]
    if n == 1:
        return np.array([A[0, 0]]), np.array([[1]])
    X = A
    Q = np.identity(n)
    sigma = 0
    for k in range(maxn):
        sigma = shift(X)
        Qi, Ri = qr(X - sigma*np.identity(n), mode='full')
        X = Ri@Qi + sigma*np.identity(n)
        Q = Q@Qi

        if norm(X[-1, :-1]) <= tol:
            T_hat, U_hat = eigh_by_QR(X[:n-1, :n-1])
            U = np.zeros((n, n))
            U[:n-1, :n-1] = U_hat
            U[-1, -1] = 1
            Q = Q@U
            T = np.append(T_hat, X[-1, -1])
            break
    idx = np.argsort(T)[::-1][:n]
    return T[idx], Q[:, idx]


def svd_phaseIIA(B: np.ndarray, Q: np.ndarray, P: np.ndarray, eigenTest=False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if eigenTest:
        T, S = scipy.linalg.eigh(B.T@B)
        T, G = scipy.linalg.eigh(B@B.T)
    else:
        T, S = eigh_by_QR(B.T@B)
        T, G = eigh_by_QR(B@B.T)

    U = G.T @ Q.T
    Vt = P @ S
    return U, T**.5, Vt


def svd(A: np.ndarray, phaseII=svd_phaseIIA, eigenTest=False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    B, Q, P = svd_phaseI(A)
    return phaseII(B, Q, P, eigenTest=eigenTest)


if __name__ == "__main__":

    print("Test HouseHolder:")
    A = np.array([[1, 0, 1],
                  [2, 5**.5, 0],
                  [0, 0, 1],
                  [0, 0, 1]])
    H = HouseHolder(A[:, 0])

    print(svd_phaseI(A))

    print("Test QR algorithm:")
    # a = np.array([[2, 1, 2],
    #               [1, 3, 3],
    #               [2, 3, 4]])
    a = A@A.T

    T, Q = eigh_by_QR(a)
    print("My Results:\nEigenvectors:\n", Q)
    print("Eigenvalues:\n", T)

    tmpE, tmpV = scipy.linalg.eigh(a)
    print("\nScipy's Results:\nEigenvectors:\n", tmpV)
    print("Eigenvalues:\n", tmpE)
