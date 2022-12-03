import numpy as np
import scipy
from scipy.linalg import norm

def Rayleigh_Quotient_Shift(A: np.ndarray) -> int:
    return A[-1, -1]


def Wilkinson_Shift(A: np.ndarray) -> int:
    T = A[-2, -2] + A[-1, -1]
    D = A[-2, -2]*A[-1, -1] - A[-1, -2]*A[-2, -1]
    e1 = T/2 + (T**2/4 - D)**.5
    e2 = T/2 - (T**2/4 - D)**.5
    return e1 if abs(e1-A[-1, -1]) < abs(e2-A[-1, -1]) else e2


def qr_tridiagonal(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """This function provide an efficient QR factorization for tridiagonal matrices using Givens Rotation.

    Args:
        T (np.ndarray): The tridiagonal matrix of interest.

    Returns:
        Q, R (np.ndarray, np.ndarray): The Q and R factors such that T = Q@R, where Q is an orthogonal 
        matrix and R is upper triangular.
    """
    Q = R = np.ndarray

    return Q, R

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