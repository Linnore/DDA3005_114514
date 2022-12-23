import numpy as np
from scipy.linalg import norm
from scipy.linalg import cholesky_banded
from scipy.linalg import cholesky

from .Bidiagonal_fastMult import *
from .QR_Factorization import applyGivenses, qr_tridiagonal_by_Givens


def Rayleigh_Quotient_Shift(A: np.ndarray) -> int:
    return A[-1, -1]


def Wilkinson_Shift(A: np.ndarray) -> int:
    T = A[-2, -2] + A[-1, -1]
    D = A[-2, -2]*A[-1, -1] - A[-1, -2]*A[-2, -1]
    e1 = T/2 + (T**2/4 - D)**.5
    e2 = T/2 - (T**2/4 - D)**.5
    return e1 if abs(e1-A[-1, -1]) < abs(e2-A[-1, -1]) else e2


def eigh_by_QR(A: np.ndarray, shift=Wilkinson_Shift, tol=1e-8, maxn=20, ascending=False, overwrite_A=False):
    """This function applies the QR algorithm with deflation on the symmetric matrix A
    to compute its eigenvalue decomposition A = Q@T@Q', where Q contains the eigenvectors
    and T is the diagonal matrix containing the corresponding eigenvalues.
    Args:
        A (np.ndarray): The matrix of interest. Note that it must be real and symmetrix.
        shift (function): Given the matrix A, shift(A) return an estimate of one eigenvalue as the shift.
        tol (float, optional): The torlerence for each defletion step. Defaults to 1e-15.
        maxn (int, optional): Maximum iterations at each defletion step. Defaults to 1000.
    Returns:
        T (np.ndarray): An 1d array that contains the eigenvalues of A in descending order.
        Q (np.ndarray): A 2d array (matrix) that contains the corresponding eigenvectors as columns.
    """
    if overwrite_A:
        X = A
    else:
        X = A.copy()
    n = X.shape[0]

    Q = np.identity(n)
    T = np.empty(n)
    for i in range(n, 1, -1):
        flag_explode = True
        for k in range(maxn):
            sigma = shift(X[:i, :i])
            np.fill_diagonal(X[:i, :i], X[:i, :i].diagonal() - sigma)

            # The following 3 lines are same as: X=QR; X=RQ.
            givens_ci, givens_si, Ri = qr_tridiagonal_by_Givens(
                X[:i, :i], return_Givens=True)
            applyGivenses(Ri, givens_ci, givens_si, axis=1)
            X[:i, :i] = Ri

            np.fill_diagonal(X[:i, :i], X[:i, :i].diagonal() + sigma)

            # The following 1 line is same as Q = Q@Qi.
            applyGivenses(Q[:, :i], givens_ci, givens_si, axis=1)

            if np.abs(X[i-1, i-2]) <= tol:
                flag_explode = False
                T[i-1] = X[i-1, i-1]
                break
        if flag_explode:
            print(k, norm(X[-1, :-1], ord=1))
            print("Max iter warning!")
    T[0] = X[0, 0]

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
    return np.diag(a[2, :], k=0) + np.diag(a[1, 1:], k=1)+np.diag(a[0, 2:], k=2)


def check_if_small(K: np.ndarray, tol=1e-8):
    def check_small_element(i: float):
        if np.abs(i) > tol:
            return 1
        else:
            return 0
    check_small_element_vec = np.vectorize(check_small_element)
    return check_small_element_vec(K)


def eigh_of_BBT(B: np.ndarray, tol=1e-8) -> tuple[np.ndarray, np.ndarray]:
    """This function applies the enhanced QR algorithm with deflation on tridiagonal matrix A = B@B.T
    to compute its eigenvalue decomposition A = Q@T@Q', where Q contains the eigenvectors
    and T is the diagonal matrix containing the corresponding eigenvalues.
    Args:
        B (np.ndarray): The upper bidiagonal square root matrix of matrix of interest.
        tol (float, optional): The torlerence for each defletion step. Defaults to 1e-15.
    Returns:
        T (np.ndarray): An 1d array that contains the eigenvalues of A in descending order.
        Q (np.ndarray): A 2d array (matrix) that contains the corresponding eigenvectors as columns.
    """

    n = B.shape[0]
    if n == 1:
        return np.array([B[0, 0]]), np.array([[1]])
    X = B
    Q = np.identity(n)
    T = np.empty(n)
    tmp = 0
    while True:
        tmp += 1
        # print(X.shape)
        if n == 1:
            T[0] = X[0, 0]
            break
        givens_ck, givens_sk, R_k = qr_tridiagonal_by_Givens(
            X[:n, :n], return_Givens=True)
        if n <= 4:
            L = cholesky(fastMult_lower_bidiagonal(R_k, R_k.T))
        else:
            ab = diagonal_form(fastMult_lower_bidiagonal(R_k, R_k.T))
            L = matrix_form(cholesky_banded(ab))
        X = L

        # The following 1 line is same as Q = Q@Qi.
        applyGivenses(Q[:n, :n], givens_ck, givens_sk)

        # upper_diag_abs = np.abs(np.diag(X[:n, :n], -1))
        # min_index = np.argmin(upper_diag_abs)
        # print(upper_diag_abs[min_index] , np.abs(X[n-1, n-2]))
        # print(norm(X[n-1, :n-1]) <= tol)

        if np.abs(X[n-2, n-1]) <= tol:
        # if norm(X[n-1, :n-1]) <= tol:
            T[n-1] = X[n-1, n-1]
            n -= 1
    print(tmp)
    idx = np.argsort(T)[::-1][:B.shape[0]]
    return T[idx], Q[:, idx]


def eigh_of_BBT_optional(B: np.ndarray, Q=None, T=None, tmp=None, start_flag=True, tol=1e-8) -> tuple[np.ndarray, np.ndarray]:
    """This function applies the enhanced QR algorithm with deflation on tridiagonal matrix A = B@B.T
    to compute its eigenvalue decomposition A = Q@T@Q', where Q contains the eigenvectors
    and T is the diagonal matrix containing the corresponding eigenvalues.
    Args:
        B (np.ndarray): The upper bidiagonal square root matrix of matrix of interest.
        tol (float, optional): The torlerence for each defletion step. Defaults to 1e-15.
        maxn (int, optional): Maximum iterations at each defletion step. Defaults to 1000.
    Returns:
        T (np.ndarray): An 1d array that contains the eigenvalues of A in descending order.
        Q (np.ndarray): A 2d array (matrix) that contains the corresponding eigenvectors as columns.
    """
    m, n = B.shape
    if n == 1:
        Q[0, 0] = 1
        T[0] = B[0, 0]
        return
        # return np.array([B[0, 0]]), np.array([[1]])
    X = B
    if start_flag:
        Q = np.identity(n)
        T = np.zeros(n)
        tmp = [0]
    while True:
        # Q_k, R_k = qr_tridiagonal_by_Givens(X)
        tmp[0] += 1
        
        upper_diag_abs = np.abs(np.diag(X, 1))
        min_index = np.argmin(upper_diag_abs)
        # print(upper_diag_abs[min_index])
        if upper_diag_abs[min_index] <= tol:
            # left_upper_diag = X[:min_index+1, :min_index+1]
            # right_upper_diag = X[min_index+1:, min_index+1:]
            if min_index>0:
                eigh_of_BBT_optional(
                    X[:min_index, :min_index],
                    Q[:min_index, :min_index],
                    tmp=tmp,
                    T=T[:min_index], start_flag=False)

            T[min_index] = X[min_index, min_index]
            if min_index<n-1:
                eigh_of_BBT_optional(
                    X[min_index+1:, min_index+1:],
                    Q[min_index+1:, min_index+1:],
                    tmp=tmp,
                    T=T[min_index+1:], start_flag=False)
            # U = np.zeros((n, n))
            # U[:min_index+1, :min_index+1] = U_hat_left
            # U[min_index+1:, min_index+1:] = U_hat_right
            # Q = Q@U
            # T = np.append(T_hat_left, T_hat_right)
            break
        # print(Q.shape)
        givens_ck, givens_sk, R_k = qr_tridiagonal_by_Givens(
            X, return_Givens=True)
        if n <= 4:
            L = cholesky(fastMult_lower_bidiagonal(R_k, R_k.T))
        else:
            ab = diagonal_form(fastMult_lower_bidiagonal(R_k, R_k.T))
            L = matrix_form(cholesky_banded(ab))
        X = L
        applyGivenses(Q, givens_ck, givens_sk)

    if start_flag:
        idx = np.argsort(T)[::-1][:n]
        print(tmp)
        return T[idx], Q[:, idx]

