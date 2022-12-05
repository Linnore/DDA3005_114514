import numpy as np
from scipy.linalg import norm
from scipy.linalg import cholesky_banded
from scipy.linalg import qr
from scipy.linalg import cholesky
from scipy.linalg import eig

def Rayleigh_Quotient_Shift(A: np.ndarray) -> int:
    return A[-1, -1]


def Wilkinson_Shift(A: np.ndarray) -> int:
    T = A[-2, -2] + A[-1, -1]
    D = A[-2, -2]*A[-1, -1] - A[-1, -2]*A[-2, -1]
    e1 = T/2 + (T**2/4 - D)**.5
    e2 = T/2 - (T**2/4 - D)**.5
    return e1 if abs(e1-A[-1, -1]) < abs(e2-A[-1, -1]) else e2


def qr_tridiagonal(T: np.ndarray, tol=1e-15, **arg) -> tuple[np.ndarray, np.ndarray]:
    """This function provide an efficient QR factorization for tridiagonal matrices using Givens Rotation.

    Args:
        T (np.ndarray): The tridiagonal matrix of interest.

    Returns:
        Q, R (np.ndarray, np.ndarray): The Q and R factors such that T = Q@R, where Q is an orthogonal 
        matrix and R is upper triangular.
    """
    X = np.array(T, dtype=float)
    m, n = X.shape
    Qt = np.identity(m)
    for i in range(n-1):
        ai = X[i, i]
        ak = X[i+1, i]
        if abs(ai) < tol and abs(ak) < tol:
            continue
        c = ai/(ai**2 + ak**2)**.5
        s = ak/(ai**2 + ak**2)**.5
        # Givens rotation
        tmp = c*X[i] + s*X[i+1]
        X[i+1] = c*X[i+1] - s*X[i]
        X[i] = tmp
        tmp = c*Qt[i] + s*Qt[i+1]
        Qt[i+1] = c*Qt[i+1] - s*Qt[i]
        Qt[i] = tmp

    return Qt.T, X


def eigh_by_QR(A: np.ndarray, shift=Wilkinson_Shift, qr=qr_tridiagonal, tol=1e-15, maxn=10000, ascending=False) -> tuple[np.ndarray, np.ndarray]:
    """This function applies the QR algorithm with deflation on the symmetric matrix A 
    to compute its eigenvalue decomposition A = Q@T@Q', where Q contains the eigenvectors
    and T is the diagonal matrix containing the corresponding eigenvalues.

    Args:
        A (np.ndarray): The matrix of interest. Note that it must be real and symmetrix.
        shift (function): Given the matrix A, shift(A) return an estimate of one eigenvalue as the shift.
        qr (function): An QR decomposition function qr(A) that returns Q and R factors given a matrix A. Defaults to scipy.linalg.qr.
        tol (float, optional): The torlerence for each defletion step. Defaults to 1e-15.
        maxn (int, optional): Maximum iterations at each defletion step. Defaults to 1000.

    Returns:
        T (np.ndarray): An 1d array that contains the eigenvalues of A in descending order.
        Q (np.ndarray): A 2d array (matrix) that contains the corresponding eigenvectors as columns.
    """
    n = A.shape[0]
    X = A.copy()
    if n == 1:
        return np.array([X[0, 0]]), np.array([[1]])
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
    if not ascending:
        idx = np.argsort(T)[::-1][:n]
    else:
        idx = np.argsort(T)
    return T[idx], Q[:, idx]


def diagonal_form(a):
    ab = np.zeros((3, a.shape[1]))
    ab[2, :] = np.diagonal(a, 0)
    ab[1, 1:] = np.diagonal(a, 1)
    ab[0, 2:] = np.diagonal(a, 2)
    return ab

def matrix_form(a):
    return np.diag(a[2, :], k=0)+ np.diag(a[1, 1:], k=1)+np.diag(a[0, 2:], k=2)

def svd_phaseIIB(B: np.ndarray, tol = 10**-12):
    res = np.zeros_like(B)
    L = B
    remaining_dim = B.shape(1)
    while remaining_dim >= 1:
        Q, R = qr_tridiagonal(B)
        ab = diagonal_form(R@R.T)                                                                                                                  
        L = matrix_form(cholesky_banded(ab))

def check_if_small(K: np.ndarray, tol = 1e-13):
    def check_small_element(i: float):
        if np.abs(i) > tol:
            return 1
        else:
            return 0
    check_small_element_vec = np.vectorize(check_small_element)
    return check_small_element_vec(K)

def eigh_by_QR_partB(B: np.ndarray, tol=1e-14, maxn=100000) -> tuple[np.ndarray, np.ndarray]:
    """This function applies the enhanced QR algorithm with deflation on tridiagonal matrix A = B.T@B by working on B
    to compute its eigenvalue decomposition A = Q@T@Q', where Q contains the eigenvectors
    and T is the diagonal matrix containing the corresponding eigenvalues.

    Args:
        B (np.ndarray): The bidiagonal square root matrix of matrix of interest.
        tol (float, optional): The torlerence for each defletion step. Defaults to 1e-15.
        maxn (int, optional): Maximum iterations at each defletion step. Defaults to 1000.

    Returns:
        T (np.ndarray): An 1d array that contains the eigenvalues of A in descending order.
        Q (np.ndarray): A 2d array (matrix) that contains the corresponding eigenvectors as columns.
    """
    n = B.shape[0]
    if n == 1:
        return np.array([B[0, 0]]), np.array([[1]])
    X = B
    Q = np.identity(n)
    for k in range(maxn):
        Q_k, R_k = qr_tridiagonal(X)
        if n <= 4:
            L = cholesky(R_k @ R_k.T)
        else:
            ab = diagonal_form(R_k@R_k.T)  
            L = matrix_form(cholesky_banded(ab))
        X = L
        Q = Q @ Q_k
        if norm(X[-1, :-1]) <= tol:
            T_hat, U_hat = eigh_by_QR_partB(X[:n-1, :n-1])
            U = np.zeros((n, n))
            U[:n-1, :n-1] = U_hat
            U[-1, -1] = 1
            Q = Q@U
            T = np.append(T_hat, X[-1, -1])
            break
    idx = np.argsort(T)[::-1][:n]
    return T[idx], Q[:, idx]

def eigh_by_QR_partB_optional(B: np.ndarray, tol=1e-14, maxn=10000) -> tuple[np.ndarray, np.ndarray]:
    """This function applies the enhanced QR algorithm with deflation on tridiagonal matrix A = B.T@B by working on B
    to compute its eigenvalue decomposition A = Q@T@Q', where Q contains the eigenvectors
    and T is the diagonal matrix containing the corresponding eigenvalues.

    Args:
        B (np.ndarray): The bidiagonal square root matrix of matrix of interest.
        tol (float, optional): The torlerence for each defletion step. Defaults to 1e-15.
        maxn (int, optional): Maximum iterations at each defletion step. Defaults to 1000.

    Returns:
        T (np.ndarray): An 1d array that contains the eigenvalues of A in descending order.
        Q (np.ndarray): A 2d array (matrix) that contains the corresponding eigenvectors as columns.
    """
    n = B.shape[0]
    if n == 1:
        return np.array([B[0, 0]]), np.array([[1]])
    X = B
    Q = np.identity(n)
    for k in range(maxn):
        Q_k, R_k = qr_tridiagonal(X)
        ab = diagonal_form(R_k@R_k.T)  
        L = matrix_form(cholesky_banded(ab))
        X = L
        Q = Q @ Q_k
        upper_diag_abs = np.abs(np.diag(X, 1))
        min_index = np.argmin(upper_diag_abs)
        if upper_diag_abs[min_index] <= tol:
            left_upper_diag = X[:min_index+1, :min_index+1]
            right_upper_diag = X[min_index+1:, min_index+1:]
            T_hat_left, U_hat_left = eigh_by_QR_partB_optional(left_upper_diag)
            T_hat_right, U_hat_right = eigh_by_QR_partB_optional(right_upper_diag)
            U = np.zeros((n, n))
            U[:min_index+1, :min_index+1] = U_hat_left
            U[min_index+1:, min_index+1:] = U_hat_right
            Q = Q@U
            T = np.append(T_hat_left, T_hat_right)
            break
    idx = np.argsort(T)[::-1][:n]
    return T[idx], Q[:, idx]