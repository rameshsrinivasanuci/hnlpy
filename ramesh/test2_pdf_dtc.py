#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:43:27 2020

@author: ramesh
"""
#%%
# import python modules
from pymatreader import read_mat
import numpy as np
import scipy.signal as signal
from scipy import linalg
from scipy.io import savemat
from matplotlib import pyplot as plt
from scipy.fftpack import fft
#%%
#import lab modules
import pdc_dtf 

# example from the paper
A = np.zeros((3, 5, 5)) #lag, channel, channel
#NOTE THAT IN THIS CODE, LAG 0 is actually one time step back. 
#THE VAR MODEL DOES NOT MAKE ANY USE OF LAG 0 CORRELATION. 
A[0, 0, 0] = 0.95 * np.sqrt(2) 
A[1, 0, 0] = -0.9025
#A[1, 1, 0] =  0.5
A[2, 2, 0] = -0.4
A[1, 3, 0] = -0.5
A[1, 3, 1] = 0.5
A[0, 3, 3] = 0.25 * np.sqrt(2)
A[0, 3, 4] = 0.25 * np.sqrt(2)
A[0, 4, 3] = -0.25 * np.sqrt(2)
A[0, 4, 4] = 0.25 * np.sqrt(2)

# simulate processes
n = 10 ** 4
# sigma = np.array([0.0001, 1, 1, 1, 1])
# sigma = np.array([0.01, 1, 1, 1, 1])
sigma = np.array([1., 1., 1., 1., 1.])
Y = pdc_dtf.mvar_generate(A, n, sigma)
#%%
mu = np.mean(Y, axis=1)
X = Y - mu[:, None]

# estimate AR order with BIC
p_max = 20
p, bic = pdc_dtf.compute_order(X, p_max=p_max)

plt.figure()
plt.plot(np.arange(p_max + 1), bic)
plt.xlabel('order')
plt.ylabel('BIC')
plt.show()

A_est, sigma = pdc_dtf.mvar_fit(X, p)
sigma = np.diag(sigma)  # DTF + PDC support diagonal noise
# sigma = None

# compute DTF
D, freqs = pdc_dtf.DTF(A_est, sigma)
pdc_dtf.plot_all(freqs, D, 'DTF')

# compute PDC
P, freqs = pdc_dtf.PDC(A_est, sigma)
pdc_dtf.plot_all(freqs, P, 'PDC')
plt.show()
#%%
#Granger Causality? 
A_estprime,sigmaprime = pdc_dtf.mvar_fit(X[[0,2,3,4],:], p)
