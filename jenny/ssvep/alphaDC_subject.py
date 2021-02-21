from pymatreader import read_mat
import numpy as np
import scipy.signal as signal
from scipy import linalg
from scipy.io import savemat
from matplotlib import pyplot as plt
from scipy.fftpack import fft2
from scipy.fftpack import fft
from scipy import signal
import os
from pymatreader import read_mat
from hdf5storage import savemat
from hdf5storage import loadmat
#%%
# import lab modules
import timeop
from ssvep_subject import *

path = '/home/jenny/pdmattention/'
subIDs = choose_subs(1, path + 'task3')
subIDs.remove('s236_ses1_')

ind = np.arange(0,33,3)
timev = np.arange(-500, 1250, 10)
title = ['theta', 'low alpha', 'high alpha', 'low beta', 'high beta']

def subjAlpha(subID):
    spec = loadmat(path + 'task3/'+'spectrogram/' + subID + 'spectrogram.mat')
    frequencies=spec['frequencies'].astype(int)
    condition = spec['condition']
    correct = spec['correct']
    goodchans = spec['goodchannels']
    goodtrials = spec['goodtrials']
    alphaDC = np.load(path + 'alphaDC/'+'desynch_%s'%subID[0:5] + '.npy')
    alphaDC2 = alphaDC[:,:,goodchans,:]
    alphaDC3 = alphaDC2[:,:,:,goodtrials]
    return alphaDC3

# timev = np.arange(-500, 1250, 10)
# fig, ax = plt.subplots(1)
# title = ['theta(3-5Hz)', 'low alpha(7-9Hz)', 'high alpha(11-13Hz)', 'low beta(15-21Hz)', 'high beta(23-29Hz)']
# for i in range(alphaDC3.shape[0]):
#     ax.plot(timev, np.mean(alphaDC3[i,75:250,:,:], axis = (1,2)).T, label = title[i])
#     ax.legend()

for i in range(0, 11):
    subID = subIDs[ind[i]:ind[i] + 3]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    count = 0
    for sub in subID:
        subalpha = subjAlpha(sub)
        for j in range(subalpha.shape[0]):
            ax[count].plot(timev, np.mean(subalpha[j, 75:250, :, :], axis=(1, 2)).T, label=title[j])
            ax[count].legend()
            ax[count].set_title('%s' % sub[0:5])
        count += 1
    fig.savefig(path + 'alphaDC/plots/%s' % sub[0:5])

for i in range(0, 11):
    subID = subIDs[ind[i]:ind[i] + 3]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    count = 0
    for sub in subID:
        subalpha = subjAlpha(sub)
        for j in range(subalpha.shape[0]):
            ax[count].plot(timev, np.mean(subalpha[j, 75:250, :, :], axis=(1, 2)).T, label=title[j])
            ax[count].legend()
            ax[count].set_title('%s' % sub[0:5])
        count += 1
    fig.savefig(path + 'alphaDC/plots/%s' % sub[0:5])

# loop thourgh each subjects and generate plots for each condition
for sub in subIDs:
    _, conditionpower, rtpower, accpower, accrtpower = alphadc(sub)
    plotcond(subject_bycond)
    plotrt(subject_rt)
    plotacc(subject_acc)
    plotaccrt(subject_accrt)

    # # by condition
    # conditionpower = np.zeros((nt, ncond, nband))
    # for k in range(ncond):
    #     trials = np.intersect1d(np.where(condition == k+1)[0],goodtrials)
    #     alpha = np.mean(alpha2[:,:,:,trials], axis=(2,3))
    #     conditionpower[:,k,:] = alpha.T
    #
    # # by RT
    # rtpower = np.zeros((nt, ncond, nband))
    # for k in range(3):
    #     npercond = np.floor(len(goodtrials)/3)
    #     trials = [j for i, j in enumerate(goodtrials) if i in np.arange(npercond*k,npercond*k+npercond,1)]
    #     alpha = np.mean(alpha2[:,:,:,trials], axis=(2,3))
    #     rtpower[:, k, :] = alpha.T
    #
    # # by accruacy
    # accpower = np.zeros((nt,2,nband))
    # for k in range(0,2):
    #     trials = np.intersect1d(np.where(correct == k)[0], goodtrials)
    #     alpha = np.mean(alpha2[:,:,:,trials], axis=(2,3))
    #     accpower[:,k,:] = alpha.T
    #
    # # by four condition
    # accrtpower = np.zeros((nt,4,nband))
    # nhalf = np.floor(len(goodtrials)/2)
    # fasttrials = goodtrials[0:int(nhalf)]
    # slowtrials = goodtrials[int(nhalf):]
    # for k in range(0,2):
    #     trials = np.intersect1d(np.where(correct == k)[0], fasttrials)
    #     alpha = np.mean(alpha2[:,:,:,trials], axis=(2,3))
    #     accrtpower[:,k,:] = alpha.T
    # for k in range(0, 2):
    #     trials = np.intersect1d(np.where(correct == k)[0], slowtrials)
    #     alpha = np.mean(alpha2[:, :, :, trials], axis=(2, 3))
    #     accrtpower[:, k+2, :] = alpha.T
