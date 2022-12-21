import numpy as np
from scipy.linalg import norm
from scipy.linalg import cholesky_banded
from scipy.linalg import cholesky
np.set_printoptions(precision=4)

def Rayleigh_Quotient_Shift(A: np.ndarray) -> int:
    return A[-1, -1]


def Wilkinson_Shift(A: np.ndarray) -> int:
    T = A[-2, -2] + A[-1, -1]
    D = A[-2, -2]*A[-1, -1] - A[-1, -2]*A[-2, -1]
    e1 = T/2 + (T**2/4 - D)**.5
    e2 = T/2 - (T**2/4 - D)**.5
    return e1 if abs(e1-A[-1, -1]) < abs(e2-A[-1, -1]) else e2


def qr_tridiagonal(T: np.ndarray, tol=1e-8, overwrite_T=False, **arg) -> tuple[np.ndarray, np.ndarray]:
    """This function provide an efficient QR factorization for tridiagonal matrices using Givens Rotation.
    Args:
        T (np.ndarray): The tridiagonal matrix of interest.
    Returns:
        Q, R (np.ndarray, np.ndarray): The Q and R factors such that T = Q@R, where Q is an orthogonal
        matrix and R is upper triangular.
    """
    if overwrite_T:
        X = T
    else:
        X = T.copy()
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

    if overwrite_T:
        return Qt.T
    else:
        return Qt.T, X


def eigh_by_QR(A: np.ndarray, shift=Wilkinson_Shift, qr=qr_tridiagonal, tol=1e-8, maxn=20, ascending=False, overwrite_A=False):
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
    if overwrite_A:
        X = A
    else:
        X = A.copy()
    n = X.shape[0]

    Q = np.identity(n)
    T = np.zeros(n)
    for i in range(n, 1, -1):
        flag_explode = True
        for k in range(maxn):
            sigma = shift(X[:i, :i])
            np.fill_diagonal(X[:i, :i], X[:i, :i].diagonal() - sigma)
            Qi, Ri = qr(X[:i, :i], mode='full')
            X[:i, :i] = Ri@Qi
            np.fill_diagonal(X[:i, :i], X[:i, :i].diagonal() + sigma)
            Q[:, :i] = Q[:, :i]@Qi

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


def eigh_of_BBT(B: np.ndarray, tol=1e-8, maxn=50) -> tuple[np.ndarray, np.ndarray]:
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


    # todo: B@B.T

    n = B.shape[0]
    if n == 1:
        return np.array([B[0, 0]]), np.array([[1]])
    X = B
    Q = np.identity(n)
    T = None
    for k in range(maxn):
        if n == 0:
            break
        # todo: qr_lower_bidiagonal
        Q_k, R_k = qr_tridiagonal(X[:n, :n])
        if n <= 4:
            L = cholesky(R_k @ R_k.T)
        else:
            ab = diagonal_form(R_k@R_k.T)
            L = matrix_form(cholesky_banded(ab))
        X = L
        Q[:n, :n] = Q[:n, :n] @ Q_k
        if norm(X[n-1, :n-1]) <= tol:
            if n == B.shape[0]:
                T = np.array(X[n-1, n-1])
            else:
                T = np.append(T, X[n-1, n-1])
            maxn = 20
            n -= 1
    idx = np.argsort(T)[::-1][:B.shape[0]]
    return T[idx], Q[:, idx]


def eigh_of_BBT_optional(B: np.ndarray, tol=1e-8, maxn=2000) -> tuple[np.ndarray, np.ndarray]:
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
            T_hat_left, U_hat_left = eigh_of_BBT_optional(left_upper_diag)
            T_hat_right, U_hat_right = eigh_of_BBT_optional(
                right_upper_diag)
            U = np.zeros((n, n))
            U[:min_index+1, :min_index+1] = U_hat_left
            U[min_index+1:, min_index+1:] = U_hat_right
            Q = Q@U
            T = np.append(T_hat_left, T_hat_right)
            break
    idx = np.argsort(T)[::-1][:n]
    return T[idx], Q[:, idx]


# Recursive way:

# def eigh_by_QR(A: np.ndarray, shift=Wilkinson_Shift, qr=qr_tridiagonal, tol=1e-8, maxn=20, ascending=False, overwrite_A=False) -> tuple[np.ndarray, np.ndarray]:
#     """This function applies the QR algorithm with deflation on the symmetric matrix A
#     to compute its eigenvalue decomposition A = Q@T@Q', where Q contains the eigenvectors
#     and T is the diagonal matrix containing the corresponding eigenvalues.
#     Args:
#         A (np.ndarray): The matrix of interest. Note that it must be real and symmetrix.
#         shift (function): Given the matrix A, shift(A) return an estimate of one eigenvalue as the shift.
#         qr (function): An QR decomposition function qr(A) that returns Q and R factors given a matrix A. Defaults to scipy.linalg.qr.
#         tol (float, optional): The torlerence for each defletion step. Defaults to 1e-15.
#         maxn (int, optional): Maximum iterations at each defletion step. Defaults to 1000.
#     Returns:
#         T (np.ndarray): An 1d array that contains the eigenvalues of A in descending order.
#         Q (np.ndarray): A 2d array (matrix) that contains the corresponding eigenvectors as columns.
#     """
#     n = A.shape[0]
#     if n == 1:
#         return np.array([[1]]), [A[0, 0]]
#     if overwrite_A:
#         X = A
#     else:
#         X = A.copy()
#     Q = np.identity(n)
#     sigma = 0
#     for k in range(maxn):
#         sigma = shift(X)
#         np.fill_diagonal(X, X.diagonal() - sigma)
#         Qi, Ri = qr(X, mode='full')
#         X = Ri@Qi
#         np.fill_diagonal(X, X.diagonal() + sigma)
#         Q = Q@Qi

#         if norm(X[-1, :-1], ord=1) <= tol:
#             del Qi
#             U_hat, T = eigh_by_QR(X[:n-1, :n-1], overwrite_A=True)
#             Q[:, :-1] = Q[:, :-1]@U_hat
#             T.append(X[-1, -1])
#             break

#     if k == maxn-1:
#         print(k, norm(X[-1, :-1], ord=1))
#         print("Max iter warning")

#     if overwrite_A:
#         return Q, T
#     else:
#         T = np.array(T)
#         if not ascending:
#             idx = np.argsort(T)[::-1][:n]
#         else:
#             idx = np.argsort(T)
#         return T[idx], Q[:, idx]