"""This file implement the two-phase singular value decomposition algorithm.
"""

import numpy as np


def hello(test):
    print(test)


def HouseHolder(a: np.ndarray) -> np.ndarray:
    """This function implement the HouseHolder transformation on a given vector 'a'.

    Args:
        a (np.ndarray): The vector of interest. 

    Returns:
        H (np.ndarray): The HouseHolder reflection matrix H with Ha = [alpha, 0, ..., 0], where abs(alpha) is the L2 norm of vector 'a'.
    """
    alpha = np.linalg.norm(a)
    n = a.shape[0]

    # To minuate cancellation effects
    if (a[0] > 0):
        alpha = -alpha

    e1 = np.zeros(n)
    e1[0] = 1
    v = a - alpha*e1
    H = np.identity(n) - 2 * np.outer(v, v) / (v.T @ v)

    return H


def svd_phaseI(A: np.ndarray) -> np.ndarray:
    """This function implement the Golub-Kahan bidiagonalization to reduce the matrix A to bidiagonal form.

    Args:
        A (np.ndarray): The matrix A whose SVD is of our interest.

    Returns:
        B (np.ndarray): The bidiagonalized version of A, with dimension min(m,n) x min(m, n).
    """
    B = np.array(A.shape)
    m, n = A.shape
    flag_T = False
    if m < n:
        flag_T = True
        A = A.T

    for i in range(n-1):
        Ut = HouseHolder(A[i:, i])
        A[i:, i:] = Ut @ A[i:, i:]

        Vi = np.zeros((n-i, n-i))
        Vi[0, 0] = 1
        Vi[1:, 1:] = HouseHolder(A[i, (i+1):].T)
        A[i:, i:] = A[i:, i:] @ Vi

    return A.T[:m, :m] if flag_T else A[:n, :n]


if __name__ == "__main__":

    print("test\n")
    A = np.array([[1, 0, 1],
                  [2, 5**.5, 0],
                  [0, 0, 1],
                  [0, 0, 1]])
    H = HouseHolder(A[:, 0])

    print(svd_phaseI(A))
