import numpy as np


def HouseHolder(a: np.ndarray, tol=1e-15) -> np.ndarray:
    """This function implement the HouseHolder transformation on a given vector 'a'.

    Args:
        a (np.ndarray): The vector of interest. 

    Returns:
        H (np.ndarray): The HouseHolder reflection matrix H with Ha = [alpha, 0, ..., 0], where abs(alpha) is the L2 norm of vector 'a'.
    """
    alpha = np.linalg.norm(a)
    n = a.shape[0]
    # if abs(alpha) < tol:
    #     # todo: permutation
    #     return np.identity(n)


    # To minuate cancellation effects
    if (a[0] > 0):
        alpha = -alpha

    e1 = np.zeros(n)
    e1[0] = 1
    v = a - alpha*e1
    H = np.identity(n) - 2 * np.outer(v, v) / (v.T @ v)

    return H
