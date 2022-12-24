import numpy as np
from numpy.linalg import norm


def HouseHolder(a: np.ndarray, tol=1e-15) -> np.ndarray:
    """This function implement the HouseHolder transformation on a given vector 'a'.

    Args:
        a (np.ndarray): The vector of interest. 

    Returns:
        w (np.ndarray): The HouseHolder reflection vector w=sqrt(2)/norm(v) * v, where v=a - alpha*e1 that define 
            H = I - 2(vv')/(v'v) with Ha = [alpha, 0, ..., 0] and abs(alpha) being the L2 norm of vector 'a'.
        alpha (float): alpha = -sign(a[0])*norm(a)
    """
    alpha = np.linalg.norm(a)
    n = a.shape[0]
    if abs(alpha) < tol:
        return np.zeros(n), 0

    # To minuate cancellation effects
    if (a[0] > 0):
        alpha = -alpha

    e1 = np.zeros(n)
    e1[0] = 1
    v = a - alpha*e1

    return np.sqrt(2)/norm(v) * v, alpha


def HouseHolder_update(A: np.ndarray, w: np.ndarray, alpha: float):

    A[0, 0] = alpha
    if A.shape[0] > 1:
        A[1:, 0] = 0
    if A.shape[1] > 1:
        w = w.reshape((-1, 1))
        A[:, 1:] = A[:, 1:] - w @ (w.T @ A[:, 1:])
    return


def GivensRotate(X: np.ndarray, c: float, s: float, axis=0):
    """This function performs in-place Givens Rotation on an ndarray X with
    shape 2xn or nx2.

    Args:
        X (np.ndarray): The ndarray to be rotate.
        c (float): _description_
        s (float): _description_
        axis (int, optional): If axis=0, the rotation will be rowwise on X, and 
        X must be in the shape 2 x n; if axis=1, then the rotation will be columnwise
        on X and X must be in the shape m x 2. Defaults to 0.
    """
    if axis == 1:
        X = X.T
    tmp = c*X[0] + s*X[1]
    X[1] = c*X[1] - s*X[0]
    X[0] = tmp


def applyGivenses(X: np.ndarray, c: np.ndarray, s: np.ndarray, axis=0):
    """This function applys a series of Givens Rotations in-place on an ndarray X.

    Args:
        X (np.ndarray): The ndarray to apply these Givens rotations.
        c, s(np.ndarray): The ndarrays that defines the series of Givens rotations. Each
            Givens rotation is defined by a matrix Gi where i=0 to n-2, Gi[i,i] = c[i], Gi[i,i+1] = s[i],
            Gi[i+1,i] = -s[i], Gi[i+1,i+1] = c[i] and remaining parts of Gi are identity matrices. 

        axis (int, optional): If axis = 0, these Givens rotations will be performed on X rowwise, i.e. X will
            be overwirtten as X=G_{n-2}@ ... @ G1 @ G0 @ X; if axis = 1, these Givens rotations will be performed 
            on X columnwise, i.e. X will be overwirtten as X = X @ G0 @ G1 @ ... @ G_{n-2}. Defaults to 0.
    """
    if axis == 1:
        X = X.T
    n = c.size
    for i in range(n):
        GivensRotate(X[i:i+2], c[i], s[i])


def qr_tridiagonal_by_Givens(T: np.ndarray, less_as_zero=1e-15, return_Givens=False) -> tuple[np.ndarray, np.ndarray]:
    """This function provide an efficient QR factorization for tridiagonal matrices using Givens Rotation.
    Comlexity of this function, wlog we assume T is n by n: 
        return_Givens = true, O(n) + O(n^2);
        return_Givens = false, O(n^2) + O(n^2);
    where the first term above depends on whether forming Q or not, 
        and the second term is for forming R factor.
    
    Args:
        T (np.ndarray): The tridiagonal matrix of interest with size m x n.
        return_Givens (bool): If this is true, then this function will form the Q factor and return Q directly;
            otherwise, this function will return the list of values c and s that define the n-1 times of 
            Givens rotations. Say each Givens rotations define a matrix Gi, i=0 to n-2, then Gi[i,i] = c[i], Gi[i,i+1] = s[i],
            Gi[i+1,i] = -s[i], Gi[i+1,i+1] = c[i] and remaining parts of Gi are identity matrices. To form
            Q, one can simply multiplay all Gi as G0 @ G1 @ ... @ G_{n-2}.
        less_as_zero (float): The tolerance for judging zero.
    Returns:
        Q, R (np.ndarray, np.ndarray): The Q and R factors such that T = Q@R, where Q is an orthogonal
        matrix and R is upper triangular.
    """
    X = np.array(T, dtype=np.float64)
    m, n = X.shape

    if return_Givens:
        givens_c = np.zeros(n-1)
        givens_s = np.zeros(n-1)
    else:
        Qt = np.identity(m)
    for i in range(n-1):
        ai = X[i, i]
        ak = X[i+1, i]
        # if np.abs(ak) < less_as_zero:
        if abs(ai) < less_as_zero and abs(ak) < less_as_zero:
            continue
        c = ai/(ai**2 + ak**2)**.5
        s = ak/(ai**2 + ak**2)**.5
        # Givens rotation
        GivensRotate(X[i:i+2], c, s)
        if return_Givens:
            givens_c[i] = c
            givens_s[i] = s
        else:
            GivensRotate(Qt[i:i+2], c, s)

    if return_Givens:
        return givens_c, givens_s, X
    else:
        return Qt.T, X

def qr_lower_bidiagonal_by_Givens(LB: np.ndarray, less_as_zero=1e-15, return_Givens=False) -> tuple[np.ndarray, np.ndarray]:
    """This function provide an efficient QR factorization for lower bidiagonal matrices using Givens Rotation.
    Comlexity of this function, wlog we assume T is n by n: 
        return_Givens = true, O(n) + O(n);
        return_Givens = false, O(n^2) + O(n);
    where the first term above depends on whether forming Q or not, 
        and the second term is for forming R factor.
    Args:
        T (np.ndarray): The tridiagonal matrix of interest with size m x n.
        return_Givens (bool): If this is true, then this function will form the Q factor and return Q directly;
            otherwise, this function will return the list of values c and s that define the n-1 times of 
            Givens rotations. Say each Givens rotations define a matrix Gi, i=0 to n-2, then Gi[i,i] = c[i], Gi[i,i+1] = s[i],
            Gi[i+1,i] = -s[i], Gi[i+1,i+1] = c[i] and remaining parts of Gi are identity matrices. To form
            Q, one can simply multiplay all Gi as G0 @ G1 @ ... @ G_{n-2}.
        less_as_zero (float): The tolerance for judging zero.
    Returns:
        Q, R (np.ndarray, np.ndarray): The Q and R factors such that T = Q@R, where Q is an orthogonal
        matrix and R is upper triangular.
    """
    X = np.array(LB, dtype=np.float64)
    m, n = X.shape

    if return_Givens:
        givens_c = np.zeros(n-1)
        givens_s = np.zeros(n-1)
    else:
        Qt = np.identity(m)
    for i in range(n-1):
        ai = X[i, i]
        ak = X[i+1, i]
        if abs(ai) < less_as_zero and abs(ak) < less_as_zero:
            continue
        c = ai/(ai**2 + ak**2)**.5
        s = ak/(ai**2 + ak**2)**.5
        # Givens rotation
        GivensRotate(X[i:i+2, i:i+2], c, s)
        if return_Givens:
            givens_c[i] = c
            givens_s[i] = s
        else:
            GivensRotate(Qt[i:i+2], c, s)

    if return_Givens:
        return givens_c, givens_s, X
    else:
        return Qt.T, X
