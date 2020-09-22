"""
# Created on 9/16/20 2:26 PM 2020 

# author: Jenny Sun 
"""

import numpy as np
from numpy import linalg
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy.fftpack import fft
from scipy.io import savemat
from pymatreader import read_mat
import timeop
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
import numpy.matlib
from collections import Counter
import imagesc
import preprocess_ostwald as po
import get_erp_ostwald as geo
from linepick import *

path = '/Volumes/My Passport/ostwald-data/clean-eeg-converted/ICA/'

def revise_ICA(subID):
    sr = 500
    currentSub = subID
    print('Current Subject: ', currentSub)
    filedict = read_mat(path + subID + '_' + "ICA" + '.mat')
    eeg = filedict['eeg']
    eog = filedict['eog']
    ecg = filedict['ecg']
    alltstim = filedict['alltstim']
    S = filedict['Sources']
    A = filedict['Mixing']
    W = filedict['Components']

    # pick a time window to see if it recovers
    recover = A @ np.transpose(S)

    fig,ax = plt.subplots(3,1)
    ax[0].plot(eeg[2000:6000,:])
    ax[0].set_title("original signal")
    ax[1].plot(S[2000:6000,:])
    ax[1].set_title("source signal")
    ax[2].plot(np.transpose(recover)[2000:6000,:])
    ax[2].set_title("recover signal")
    fig.set_size_inches(11.56,8.91)

    # construct a time x channel x trial matrix for each run for 5s
    samples = int(5 * sr)
    channelnum = eeg.shape[1]
    trialnum = len(alltstim)

    # component space
    trialcomponent = np.zeros((samples, channelnum, trialnum))

    # raw eeg space
    trialeeg = np.zeros((samples, channelnum, trialnum))

    # epoch the data to create single-trial segments for 5s
    for i in np.arange(trialnum):
        time = alltstim[i]
<<<<<<< HEAD
        trialeeg[:, :, i] = eeg[time - 1000: time + 1500, :]

    getline(np.mean(trialcomponent, axis=2))
    A[:, 24] = 0


    # plot the raw erp
    cleaneeg = np.zeros((samples, channelnum, trialnum))
=======
        trialcomponent[:, :, i] = S[time - 1000: time + 1500, :]
>>>>>>> updated ica2

    # epoch the data to create single-trial segments for 5s
    for i in np.arange(trialnum):
        time = alltstim[i]
        trialeeg[:, :, i] = eeg[time - 1000: time + 1500, :]

    # baseline correction for component ERP
    trialcomponent_corr = np.zeros((samples, channelnum, trialnum))
    trialeeg_corr = np.zeros((samples, channelnum, trialnum))

    # component space
    for i in range(0, trialnum):
        baseline_mean1 = np.tile(np.mean(trialcomponent[(500 - 50):500, :, i], axis=0), [950, 1])
        trialcomponent_corr[0:950, :, i] = trialcomponent[0:950, :, i] - baseline_mean1
        baseline_mean2 = np.tile(np.mean(trialcomponent[(1000 - 50):1000, :, i], axis=0), [1550, 1])
        trialcomponent_corr[950:2500, :, i] = trialcomponent[950:2500, :, i] - baseline_mean2

    # raw eeg space
    for i in range(0, trialnum):
        baseline_mean1 = np.tile(np.mean(trialeeg[(500 - 50):500, :, i], axis=0), [950, 1])
        trialeeg_corr[0:950, :, i] = trialeeg[0:950, :, i] - baseline_mean1
        baseline_mean2 = np.tile(np.mean(trialeeg[(1000 - 50):1000, :, i], axis=0), [1550, 1])
        trialeeg_corr[950:2500, :, i] = trialeeg[950:2500, :, i] - baseline_mean2

    # plot the component ERP and the eeg ERP
    fig,ax = plt.subplots(2,1)
    ax[0].plot(np.arange(-100,2000,2), np.mean(trialcomponent_corr[450:1500], axis = 2))
    ax[0].axvline(1000)
    ax[0].axvline(0)
    ax[0].set_title("erp in the component space")

    ax[1].plot(np.arange(-100,2000,2), np.mean(trialeeg_corr[450:1500], axis = 2))
    ax[1].set_title("erp from the raw data")
    ax[1].axvline(1000)
    ax[1].axvline(0)

    # identify the artifact components by calling linepick module
    getline(np.arange(0, 5000, 2), np.mean(trialcomponent_corr, axis=2))
    plt.locator_params(nbins=50)
    plt.axvline(1000)
    plt.axvline(2000)
    # highlight the channels
    plt.plot(np.arange(0, 5000, 2), np.mean(trialcomponent_corr[:, 0, :], axis=1), linewidth=4)

    recover_clean = A @ np.transpose(S)

    # clean eeg space
    cleaneeg = np.zeros((samples, channelnum, trialnum))

    # epoch the data to create single-trial segments for 5s
    recover_clean = np.transpose(recover_clean)
    for i in np.arange(trialnum):
        time = alltstim[i]
        cleaneeg[:, :, i] = recover[time - 1000: time + 1500, :]






    # compare the component ERP and the eeg ERP
    fig,ax = plt.subplots(2,1)
    ax[0].plot(np.arange(-100,2000,2), np.mean(trialcomponent_corr[450:1500], axis = 2))
    ax[0].axvline(1000)
    ax[0].axvline(0)
    ax[0].set_title("erp in the component space")

    ax[1].plot(np.arange(-100,2000,2), np.mean(trialeeg_corr[450:1500], axis = 2))
    ax[1].set_title("erp from the raw data")
    ax[1].axvline(1000)
    ax[1].axvline(0)

