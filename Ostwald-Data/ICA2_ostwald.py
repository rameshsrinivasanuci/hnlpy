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
import linepick

base_dir = "/home/jenny/ostwald-data/"
data_dir = "/home/jenny/ostwald-data/clean-eeg-converted/"
path = '/home/jenny/ostwald-data/clean-eeg-converted/ICA/'

def rerun_ICA():
    subID, _ = geo.get_sub(path)
    for sub in subID:
        get_trialdata(sub)
        print(f'\n{sub} saved\n')

def get_trialdata(subID):
    """this function segments the data into trials into 5s time x trial x channels data
    based on the stimulus time index
    eeg is the complete time x 64 continuous data
    goodchans are the channels that exclude bad channels identified before ICA, and
    eog and ecg"""
    sr = 500
    currentSub = subID
    print('Current Subject: ', currentSub)
    filedict = read_mat(path + subID + '_' + "ica" + '.mat')
    chandict = read_mat(base_dir + 'code/' + "pdm_erp_electrode_locations.mat")
    chanlocs = np.array(chandict['chanlocs'])
    eeg = filedict['eeg']
    eog = eeg[:,30]
    ecg = eeg[:,31]
    alltstim = filedict['alltstim']
    allcond = filedict['allcond']
    S = filedict['Sources']
    A = filedict['Mixing']
    W = filedict['Components']

    # identify goodchannels
    badchans = filedict['badchans']
    channels = np.arange(0,eeg.shape[1])
    goodchans = [i for i in channels if i not in badchans]

    # pick a time window to see if it recovers
    recover = A @ np.transpose(S)

    fig,ax = plt.subplots(3,1)
    ax[0].plot(eeg[5000:9000,goodchans])
    ax[0].set_title("original signal")
    ax[1].plot(S[5000:9000,:])
    ax[1].set_title("source signal")
    ax[2].plot(np.transpose(recover)[5000:9000,:])
    ax[2].set_title("recover signal")
    fig.set_size_inches(11.56,8.91)

    # construct a time x channel x trial matrix for each run for 5s
    samples = int(5 * sr)
    channelnum = eeg.shape[1]
    componentnum= S.shape[1]
    trialnum = len(alltstim)

    # component space
    trialcomponent = np.zeros((samples, componentnum, trialnum))

    # raw eeg space
    trialeeg = np.zeros((samples, channelnum, trialnum))

    # epoch raw eeg data to create single-trial segments for 5s
    for i in np.arange(trialnum):
        time = alltstim[i]
        trialeeg[:, :, i] = eeg[time - 1000: time + 1500, :]

    # epoch components to create single-trial segments for 5s
    for i in np.arange(trialnum):
        time = alltstim[i]
        trialcomponent[:, :, i] = S[time - 1000: time + 1500, :]

    # getline(np.mean(trialcomponent, axis=2))
    # A[:, 24] = 0



    # # identify the artifact components by calling linepick module
    # getline(np.arange(-100, 700, 2), np.mean(trialeeg[950:1350], axis=2))
    # plt.locator_params(nbins=50)
    # plt.axvline(1000)
    # plt.axvline(2000)
    # # highlight the channels
    # plt.plot(np.arange(0, 5000, 2), np.mean(trialcomponent_corr[:, 0, :], axis=1), linewidth=4)
    #
    # # recover_clean = A @ np.transpose(S)
    #
    # # clean eeg space
    # cleaneeg = np.zeros((samples, channelnum, trialnum))

    # # epoch the data to create single-trial segments for 5s
    # recover_clean = np.transpose(recover_clean)
    # for i in np.arange(trialnum):
    #     time = alltstim[i]
    #     cleaneeg[:, :, i] = recover[time - 1000: time + 1500, :]
    icadict = {}
    icadict["alltstim"] = alltstim
    icadict["allcond"] = allcond
    icadict["sr"] = sr
    icadict["eeg"] = eeg
    icadict["goodchans"] = goodchans
    icadict["trialcomponent"] = trialcomponent
    icadict["trialeeg"] = trialeeg

    savemat(f'/home/jenny/ostwald-data/clean-eeg-converted/ICA/{subID}_trialdata.mat',icadict)
    return icadict

def runall():
    subID, _ = geo.get_sub(path)
    for sub in subID:
        generate_ICAplots(sub)

def generate_ICAplots(subID):
    """this function reads from the trialdata.mat files
    baseline corrects the data for generating plots"""

    currentSub = subID
    print('Current Subject: ', currentSub)
    datadict = read_mat(path + subID + '_' + "trialdata" + '.mat')
    icadict = read_mat(path + subID + '_' + "ica" + '.mat')
    eeg = datadict['eeg']
    eog = eeg[:,30]
    ecg = eeg[:,31]
    alltstim = datadict['alltstim']
    allcond = datadict['allcond']
    trialcomponent = datadict['trialcomponent']
    trialeeg = datadict['trialeeg']
    goodchans = datadict['goodchans']
    sr = datadict['sr']
    S = icadict['Sources']


    # baseline correction for component ERP
    samples = int(5 * sr)
    channelnum = trialeeg.shape[1]
    componentnum = trialcomponent.shape[1]
    trialnum = len(alltstim)

    trialcomponent_corr = np.zeros((samples, componentnum, trialnum))
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

    # compare the component ERP and the eeg ERP
    fig,ax = plt.subplots(2,1)
    ax[0].plot(np.arange(-100,2000,2), np.mean(trialcomponent_corr[450:1500], axis = 2))
    ax[0].axvline(1000)
    ax[0].axvline(0)
    ax[0].set_title("erp in the component space")

    ax[1].plot(np.arange(-100,2000,2), np.mean(trialeeg_corr[450:1500, goodchans,:], axis = 2))
    ax[1].set_title("erp from the raw data")
    ax[1].axvline(1000)
    ax[1].axvline(0)

    # get the correlation coefficient between EOG and channel from Source Matrix
    chanmatrix = np.column_stack((S, eog))
    R = np.corrcoef(np.transpose(chanmatrix))
    corr = R[-1, :]

    # find the two highest correlation channels
    corr1 = np.argmax(abs(corr[:-1]))
    corr2_val = sorted(abs(corr[:-1]))[-2]
    corr2 = [x for x, i in enumerate(corr[:-1]) if i == corr2_val or i == -1 * corr2_val]
    corr2 = int(corr2[0])

    # compare the raw eog with the highly correcltaed copmonents
    # normalize the components
    componentmax = max(max(S[0:10000, corr1]), max(S[0:10000, corr2]))
    eogmin = min(eog[0:10000])
    eogmax = max(eog[0:10000])
    eogrange = eogmax - eogmin
    neweog = ((eog[0:10000] - eogmin) / eogrange) * componentmax

    # epoch eog data
    samples = int(5 * sr)
    trialnum = len(alltstim)
    trialeog = np.zeros((samples, trialnum))

    # epoch the data to create single-trial segments for 5s
    for i in np.arange(trialnum):
        time = alltstim[i]
        trialeog[:, i] = eog[time - 1000: time + 1500]

    # compare the highly correlated components with and the eog channel

    componentmax = max(max(np.mean(trialcomponent[:, corr1, :], axis=1)),
                       max(np.mean(trialcomponent[:, corr2, :], axis=1)))
    eogmean = np.mean(trialeog, axis=1)
    eogmin = min(eogmean)
    eogmax = max(eogmean)
    eogrange = eogmax - eogmin
    neweogmean = ((eogmean - eogmin) / eogrange) * componentmax

    # plot the component ERP and the eeg ERP
    fig,ax = plt.subplots(2,1)
    ax[0].plot(np.arange(-100,700,2), np.mean(trialcomponent_corr[950:1350], axis = 2))
    ax[0].plot(np.arange(-100,700,2), np.mean(trialcomponent_corr[950:1350,corr1,:], axis = 1),color='black',linewidth = 3.5)
    ax[0].plot(np.arange(-100, 700, 2), np.mean(trialcomponent_corr[950:1350, corr2, :], axis=1),color='blue',linewidth=3.5)
    ax[0].axvline(0)
    ax[0].set_title("erp in the component space")

    ax[1].plot(np.arange(-100,700,2), np.mean(trialeeg_corr[950:1350,goodchans,:], axis = 2))
    ax[1].set_title("erp from the raw data")
    ax[1].axvline(0)
    fig.suptitle(f"{subID}")

    getline(np.arange(-100, 700, 2), np.mean(trialeeg_corr[950:1350], axis=2))


    plt.savefig((f'/home/jenny/ostwald-data/clean-eeg-converted/ICA/Figures/{subID}componentERP.png'), \
                dpi=300, format='png',bbox_inches='tight')





