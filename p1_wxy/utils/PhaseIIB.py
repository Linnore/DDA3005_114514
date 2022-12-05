import numpy as np
import scipy.linalg
from QR import qr_tridiagonal


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)
'''

Q, R = scipy.linalg.qr(A.T)
print(Q @ R)

Q, R = qr_tridiagonal(A.T)
print(Q @ R)
'''


def diagonal_form(a):
    ab = np.zeros((2, a.shape[1]))
    ab[1, :] = np.diagonal(a, 0)
    ab[0, 1:] = np.diagonal(a, 1)
    return ab

def bidiagonal_form(a):
    return np.diag(a[1, :], k=0)+ np.diag(a[0, 1:], k=1)

A = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]])

print(scipy.linalg.svd(A)[0])

'''
print(G)

print(bidiagonal_form(scipy.linalg.cholesky_banded(diagonal_form(G))))

print(scipy.linalg.cholesky(G))
'''
#scipy.linalg.cholesky_banded(diagonal_form(R @ R.T))