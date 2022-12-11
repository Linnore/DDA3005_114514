import numpy as np
import scipy
import utils
from scipy.stats import qmc
from utils import HouseHolder, QR, SVD
import scipy.io
import scipy.linalg
from PIL import Image
from numpy import asarray
from matplotlib import image
from matplotlib import pyplot as plt
import time
import os

def generate_tri_T(n,k):
    """
    Creates a 2-dimensional, size x size blurring matrix following hw3 problem4.
    Here the convoluation kernel is 2*2 kernel
    It is normalized such that the sum over all values = 1. 
    Args:
        n (int):     The dimensionality of the matrix.
        k (int):    The power of the matrix
    Returns:
        A n x n floating point ndarray whose values are sampled following hw3 problem4.
    """

    delta = 0.1
    diag_element = (2+delta)/(4+delta)
    tridiagonal_element = 1/(4+delta)
    T = np.diag([tridiagonal_element]*(n-1), k = -1)+np.diag([tridiagonal_element]*(n-1), k = 1)+np.identity(n)*(diag_element)
    T_k = np.linalg.matrix_power(T,k)
    return T_k

def generate_gaussian_T(n,k):
    """
    Creates a 2-dimensional, size x size gaussian blurring matrix.
    Here the convoluation kernel is 5*5 Gaussian kernel
    It is normalized such that the sum over all values = 1. 
    Args:
        n (int):     The dimensionality of the kernel.
        k (int):    The power of the kernel
    Returns:
        A n x n floating point ndarray whose values are sampled from the multivariate gaussian.
    """
    T = np.diag([3/8]*n, 0)+np.diag([1/4]*(n-1), 1)+np.diag([1/4]*(n-1), -1)+np.diag([1/16]*(n-2), 2)+np.diag([1/16]*(n-2), -2)
    T_k = np.linalg.matrix_power(T,k)
    return T_k

def generate_box_T(n,k):
    """
    Creates a 2-dimensional, size x size box blurring .
    Here the convoluation kernel is 2*2 box kernel
    It is normalized such that the sum over all values = 1. 
    Args:
        n (int):     The dimensionality of the matrix.
        k (int):    The power of the matrix
    Returns:
        A n x n floating point ndarray whose values are sampled.
    """
    T = np.diag([1/3]*n, 0)+np.diag([1/3]*(n-1), 1)+np.diag([1/3]*(n-1), -1)
    T_k = np.linalg.matrix_power(T,k)
    return T_k

def generate_T(n, blur_type,power):
    """
    According to the input type to generate different blur matirx.
    Args:
        n(int):     The dimensionality of the matrix. 
        blur_type(string): Different type of blurring.
                            'tridiagonal' refers to the blurring matrix with the kernel following hw3 problem4.
                            'gaussian' refers to the blurring matrix with the gaussian .
                            'box' refers to the blurring matrix with box kernel.

    Returns:
        A n x n floating point ndarray whose values are sampled.

    """
    if blur_type == 'tridiagonal':
        return generate_tri_T(n,power)
    elif blur_type == 'gaussian':
        return generate_gaussian_T(n,power)
    elif blur_type == 'box':
        return generate_box_T(n,power)
    
        
#blur the picture
def blur_picture(original_data,blur_type_l,blur_type_r,power_l, power_r):
    m,n,k = original_data.shape
    blur_data = np.zeros((m,n,k))
    blur_kernel_l = generate_T(m,blur_type_l,power_l)
    blur_kernel_r = generate_T(m,blur_type_r,power_r)
    for i in range(k):
        blur_data[:,:,i]= blur_kernel_l@original_data[:,:,i]@blur_kernel_r
    return blur_kernel_l,blur_kernel_r,blur_data

def truncated_inverse(matr,trunc,size,svd_type):
    u,sigma,v = scipy.linalg.svd(matr)
    v = v.T
    A = np.zeros((size,size))
    for i in range(0,trunc):
        A += (np.outer(v[:,i],u[:,i])/sigma[i])
    return A

def deblur_picture(blur_kernel_l,blur_kernel_r,blur_data,trunc,svd_type):
    m,n,k = blur_data.shape
    psnr = []
    deblur_data = np.zeros((m,n,k))
    A_l = truncated_inverse(blur_kernel_l,trunc,m,svd_type)
    A_r = truncated_inverse(blur_kernel_r,trunc,m,svd_type)
    for i in range(k):
        deblur_data[:,:,i]= A_l@blur_data[:,:,i]@A_r
        norm_deblur = np.linalg.norm(deblur_data[:,:,i],'fro')
        psnr.append(10*np.log10(m**2/norm_deblur**2))  
    return deblur_data,np.array(psnr).mean()


def singular_drawing(matr):
    u,sigma,v = SVD.svd(matr)
    plt.plot(sigma)
