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

    return np.sqrt(2)/norm(v) * v , alpha


def HouseHolder_update(A: np.ndarray, w: np.ndarray, alpha: float):

    A[0, 0] = alpha
    if A.shape[0] > 1:
        A[1:, 0] = 0
    if A.shape[1] > 1:
        A[:, 1:] = A[:, 1:] - w @ (w.T @ A[:, 1:])
    return


# def HouseHolder(a: np.ndarray, tol=1e-15) -> np.ndarray:
#     """This function implement the HouseHolder transformation on a given vector 'a'.

#     Args:
#         a (np.ndarray): The vector of interest. 

#     Returns:
#         H (np.ndarray): The HouseHolder reflection matrix H with Ha = [alpha, 0, ..., 0], where abs(alpha) is the L2 norm of vector 'a'.
#     """
#     alpha = np.linalg.norm(a)
#     n = a.shape[0]
#     if abs(alpha) < tol:
#         return np.identity(n)

#     # To minuate cancellation effects
#     if (a[0] > 0):
#         alpha = -alpha

#     e1 = np.zeros(n)
#     e1[0] = 1
#     v = a - alpha*e1
#     H = np.identity(n) - 2 * np.outer(v, v) / (v.T @ v)
#     return H