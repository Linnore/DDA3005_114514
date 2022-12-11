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
        n(int):             The dimensionality of the matrix. 
        blur_type(string): Different type of blurring.
                            'tridiagonal' refers to the blurring matrix with the kernel following hw3 problem4.
                            'gaussian' refers to the blurring matrix with the gaussian .
                            'box' refers to the blurring matrix with box kernel.
        power(int):         The power wanted to use for the blurring matirx.

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
def blur_picture(original_data,blur_type,power):
    """
    According to the input type to blur the origiinal matrix.
    Args:
        original_data (2d array):       The original image data      
        blur_type(list[str,str]):       Different type of blurring. 
                                        First one refers to the left blur matrix and second one refers to the right blur matrix.
                                        'tridiagonal' refers to the blurring matrix with the kernel following hw3 problem4.
                                        'gaussian' refers to the blurring matrix with the gaussian .
                                        'box' refers to the blurring matrix with box kernel.
         power(list[int,int]):         The power wanted to use for the blurring matirx.
                                        First one refers to the power for left blur matrix and second one refers to the power for right blur matrix.

    Returns:
        The list of left and right blurring matirx, blurring image data. 
    """
    m,n,k = original_data.shape
    blur_data = np.zeros((m,n,k))
    blur_kernel_l = generate_T(m,blur_type[0],power[0])
    blur_kernel_r = generate_T(m,blur_type[1],power[1])
    for i in range(k):
        blur_data[:,:,i]= blur_kernel_l@original_data[:,:,i]@blur_kernel_r
    return [blur_kernel_l,blur_kernel_r],blur_data

def truncated_inverse(matr,trunc,svd_type):
    """
    Calculate the inverse using truncated SVD.
    Args:
        matr(2d array):           The matrix needed to be calculate inverse.
        trunc(int)                The range used to calculate the truncated inverse.               
        svd_type(int):            The svd method we use to do the svd decomposition.
                                    0 refers to the Scipy svd decomposition;
                                    1 refers to the method in problem1 phaseI and IIA;
                                    2 referse to the method in porblem1 phaseI and IIB;


    Returns:
        The truncated inverse; the time used for the svd decomposition
    """
    if svd_type == 0:
        begin = time.time()
        u,sigma,v = scipy.linalg.svd(matr)
        time_used = time.time()-begin()
    if svd_type == 1:
        begin = time.time()
        u,sigma,v = SVD.svd(matr)
        time_used = time.time()-begin()
    v = v.T
    size = matr.shape[0]
    A = np.zeros((size,size))
    for i in range(0,trunc):
        A += (np.outer(v[:,i],u[:,i])/sigma[i])
    return A,time_used

def deblur_picture(blur_kernel,blur_data,trunc,svd_type):
    """
    Calculate the inverse using truncated SVD.
    Args:
        blur_kernel(list[2d array,2d array]):    The matrixes generated in the blur_picture() to blur the picture.
                                                 First one refers to the left blur matrix and second one refers to the right blur matrix.
        blur_data(3d array):                    The blurring image data generated in the blur_picture().
        trunc(list[int,int])                The list of range used to calculate the truncated inverse.           
                                            First one refers to the truncaion used for left blur matrix and second one refers to truncation used for the right blur matrix.    
        svd_type(list[int,int]):            The svd method we use to do the svd decomposition.
                                            First one refers to the svd decomposition used for left blur matrix and second one refers to svd decomposition used for the right blur matrix.    
                                            0 refers to the Scipy svd decomposition;
                                            1 refers to the method in problem1 phaseI and IIA;
                                            2 referse to the method in porblem1 phaseI and IIB;


    Returns:
        The deblur data; the average peak-signal-to-noise ratiol; the time used for the svd decomposition
    """
    m,n,k = blur_data.shape
    psnr = []
    deblur_data = np.zeros((m,n,k))
    A_l,time_l = truncated_inverse(blur_kernel[0],trunc,svd_type[0])
    A_r,time_r = truncated_inverse(blur_kernel[1],trunc,svd_type[1])
    time_svd = [time_l,time_r]
    for i in range(k):
        deblur_data[:,:,i]= A_l@blur_data[:,:,i]@A_r
        norm_deblur = np.linalg.norm(deblur_data[:,:,i],'fro')
        psnr.append(10*np.log10(m**2/norm_deblur**2))  
    return deblur_data,np.array(psnr).mean(),time_svd


def singular_drawing(matr):
    """
    Draw the plot of singular values of a matrix.
    """
    u,sigma,v = SVD.svd(matr)
    plt.plot(sigma)
