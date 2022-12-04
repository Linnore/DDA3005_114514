import numpy as np
from scipy.linalg import norm


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
        # if abs(ai) < tol and abs(ak)<tol:
        #     continue
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


def eigh_by_QR(A: np.ndarray, shift=Wilkinson_Shift, qr=qr_tridiagonal, tol=1e-15, maxn=1000, ascending=False) -> tuple[np.ndarray, np.ndarray]:
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
