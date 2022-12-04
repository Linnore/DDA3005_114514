import numpy as np
import scipy
import utils
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
    delta = 0.1
    diag_element = (2+delta)/(4+delta)
    tridiagonal_element = 1/(4+delta)
    T = np.diag([tridiagonal_element]*(n-1), k = -1)+np.diag([tridiagonal_element]*(n-1), k = 1)+np.identity(n)*(diag_element)
    T_k = np.linalg.matrix_power(T,k)
    return T_k


#blur the picture
def blur_picture(original_data,blur_type='tridiagonal',power_l=8, power_r = 8):
    m,n,k = original_data.shape
    blur_data = np.zeros((m,n,k))
    if blur_type == 'tridiagonal':
        blur_kernel_l = generate_tri_T(m,power_l)
        blur_kernel_r = generate_tri_T(m,power_r)
    for i in range(k):
        blur_data[:,:,i]= blur_kernel_l@original_data[:,:,i]@blur_kernel_r
    return blur_kernel_l,blur_kernel_r,blur_data

def truncated_inverse(matr,trunc,size):
    u,sigma,v = SVD.svd(matr)
    v = v.T
    A = np.zeros((size,size))
    for i in range(0,trunc):
        A += (np.outer(v[:,i],u[:,i])/sigma[i])
    return A

def deblur_picture(blur_kernel_l,blur_kernel_r,blur_data,trunc):
    m,n,k = blur_data.shape
    psnr = []
    deblur_data = np.zeros((m,n,k))
    A_l = truncated_inverse(blur_kernel_l,trunc,m)
    A_r = truncated_inverse(blur_kernel_r,trunc,m)
    for i in range(k):
        deblur_data[:,:,i]= A_l@blur_data[:,:,i]@A_r
        norm_deblur = np.linalg.norm(deblur_data[:,:,i],'fro')
        psnr.append(10*np.log10(m**2/norm_deblur**2))  
    return deblur_data,np.array(psnr).mean()


def singular_drawing(matr):
    u,sigma,v = SVD.svd(matr)
    plt.plot(sigma)
