"""This file contains some operations that boosted by exploiting the structure of
bidiagonal matrices. 
"""


import numpy as np


def fastMult_upper_bidiagonal(A: np.ndarray, UB: np.ndarray) -> np.ndarray:
    """This function exploit the struction of a bidiagonal matrix to compute A@UB in O(n^2), where A
    is a general matrix and UB is the upper bidiagonal matrix.

    Args:
        A (np.ndarray): _description_
        B (np.ndarray): _description_

    Returns:
        (np.ndarray): The product A@B
    """
    m, n = (A.shape[0], UB.shape[-1])
    result = np.zeros((m, n))
    result[:, 0] = UB[0, 0]*A[:, 0]
    for i in range(1, n):
        result[:, i] = UB[i-1, i]*A[:, i-1] + UB[i, i]*A[:, i]
    return result


def upper_fastMult_lower_bidiagonal(UB: np.ndarray, LB: np.ndarray) -> np.ndarray:
    """This function exploit the struction of bidiagonal matrices to compute UB@LB in O(n), 
    where UB is a upper bidiagonal matrix and LB is a lower bidiagonal matrix.

    Args:
        UB (np.ndarray): _description_
        LB (np.ndarray): _description_

    Returns:
        (np.ndarray): The product UB@LB
    """
    m, n = (UB.shape[0], LB.shape[-1])

    result = np.zeros((m, n))
    result[0, 0] = LB[0, 0] * UB[0, 0]
    result[-1, -1] = LB[-1, -1]*UB[-1, -1]
    if m >= 2:
        result[0:2, 0] += LB[1, 0] * UB[0:2, 1]
        result[-2, -1] += LB[-1, -1]*UB[-2, -1]
    for i in range(1, n-1):
        result[i-1:i+1, i] += LB[i, i] * UB[i-1:i+1, i]
        result[i:i+2, i] += LB[i+1, i] * UB[i:i+2, i+1]

    return result
