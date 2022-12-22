"""This file contains some operations that boosted by exploiting the structure of
bidiagonal matrices. 
"""


import numpy as np

def fastMult_upper_bidiagonal(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """This function exploit the struction of a bidiagonal matrix to compute A@B in O(n^2), where A
    is a general matrix and B is the upper bidiagonal matrix.

    Args:
        A (np.ndarray): _description_
        B (np.ndarray): _description_

    Returns:
        (np.ndarray): The product A@B
    """
    m, n = (A.shape[0], B.shape[-1])
    result = np.zeros((m, n))
    result[:, 0] = B[0, 0]*A[:, 0]
    for i in range(1, n):
        result[:, i] = B[i-1, i]*A[:, i-1] + B[i, i]*A[:, i]
    return result


def fastMult_lower_bidiagonal(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """This function exploit the struction of a bidiagonal matrix to compute A@B in O(n^2), where A
    is a general matrix and B is the lower bidiagonal matrix.

    Args:
        A (np.ndarray): _description_
        B (np.ndarray): _description_

    Returns:
        (np.ndarray): The product A@B
    """
    m, n = (A.shape[0], B.shape[-1])
    result = np.zeros((m, n))
    result[:, -1] = B[-1, -1]*A[:, -1]
    for i in range(n-2, -1, -1):
        result[:, i] = B[i+1, i]*A[:, i+1] + B[i, i]*A[:, i]
    return result