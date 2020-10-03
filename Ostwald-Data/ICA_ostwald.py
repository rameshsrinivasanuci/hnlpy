#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 00:56:46 2020

@author: Jenny Sun
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

path = '/home/jenny/ostwald-data/clean-eeg-converted/'


def get_data(subID, run, path):
    """
    this function reads the data from inside scanner, output dictionary contains
    'data': raw data from .mat files, with the mean removed
    'condition': one row of the conditions
    'tstim': time index for stimulus"""

    currentSub = subID
    currentRun = 'run-'+ run
    print('Current Subject: ', currentSub)
    print('Current Run:', currentRun)
    filedict = read_mat(path + subID + '_' +  currentRun + '_eeg'+'.mat')
    datadict = filedict['outputdata']
    eventsdict = (datadict['event'])

    data = np.array(datadict['data'])
    sr = np.array(datadict['srate'])
    latency = np.array(eventsdict['latency'])
    eventtype = eventsdict['type']
    code = np.array(eventsdict['code'])

    # get the index of stimulus
    left_hcp = [i for i, c in enumerate(eventtype) if 'S 10' in c]
    left_hcnp = [i for i, c in enumerate(eventtype) if 'S 20' in c]
    left_lcp = [i for i, c in enumerate(eventtype) if 'S 30' in c]
    left_lcnp = [i for i, c in enumerate(eventtype) if 'S 40' in c]

    right_hcp = [i for i, c in enumerate(eventtype) if 'S 11' in c]
    right_hcnp = [i for i, c in enumerate(eventtype) if 'S 21' in c]
    right_lcp = [i for i, c in enumerate(eventtype) if 'S 31' in c]
    right_lcnp = [i for i, c in enumerate(eventtype) if 'S 41' in c]

    cond1 = np.array((left_hcp + right_hcp,[1]*(len(right_hcp)+len(left_hcp))))
    cond2 = np.array((left_hcnp + right_hcnp, [2] * (len(left_hcnp) + len(right_hcnp))))
    cond3 = np.array((left_lcp + right_lcp, [3] * (len(left_lcp) + len(right_lcp))))
    cond4 = np.array((left_lcnp + right_lcnp, [4] * (len(left_lcnp) + len(right_lcnp))))
    condList = np.transpose(np.concatenate((cond1,cond2,cond3,cond4),axis=1))

    # get a matrix where the first column is the index, second column is the condition
    cond_ind = sorted(condList, key=lambda condList_entry: condList_entry[0])
    cond = [i[1] for i in cond_ind]

    all_stim = left_hcp + left_hcnp + left_lcp + left_lcnp\
               + right_hcp + right_hcnp + right_lcp + right_lcnp
    all_stim.sort()

    # index the latency, minus one because times series starts at 0 index
    tstim = latency[all_stim]
    tstim = tstim -1

    # transpose the data
    data = np.transpose(data)

    # # eeg = np.delete(data, slice(30, 32), axis=1)
    #
    # # construct a time x channel x trial matrix for each run for 5s
    # samples = int(5*sr)
    # channelnum = data.shape[1]
    # trialnum = tstim.shape[0]
    # trialdata = np.zeros((samples,channelnum, trialnum))
    #
    # # epoch the data to create single-trial segments for 5s
    # for i in np.arange(trialnum):
    #     time = tstim[i]
    #     trialdata[:,:, i] = data[time-1000: time+1500,:]
    #
    # # remove EOG and ECG and get trialeeg
    # trialeeg = np.delete(trialdata, slice(30, 32), axis=1)
    #
    # # baseline correction
    # # get the mean of 100ms pre-stim and remove it from the whole window
    # for i in range(0, trialnum):
    #     baseline_mean = np.tile(np.mean(trialeeg[(1000 - 50):1000, :, i], axis=0), [trialeeg.shape[0], 1])
    #     trialeeg[:, :, i] = trialeeg[:, :, i] - baseline_mean

    # remove the mean of the raw data
    channelmean = np.mean(data, axis=0)
    data = data - np.tile(channelmean, [data.shape[0], 1])

    # save to dataDict
    dataDict = {
        # 'trialeeg': trialeeg,
        # 'trialdata': trialdata,
        'data': data,
        'condition': cond,
        'tstim': tstim
        }
    return dataDict


def combine_runs(subID, path):
    """this function combines all the runs, stimulus, and condition
    allrun, tstimnall, allcond = combine_runs(subID)"""

    dataDict1 = get_data(subID, "01", path)
    dataDict2 = get_data(subID, "02", path)
    dataDict3 = get_data(subID, "03", path)
    dataDict4 = get_data(subID, "04", path)
    dataDict5 = get_data(subID, "05", path)
    data1 = dataDict1['data']
    data2 = dataDict2['data']
    data3 = dataDict3['data']
    data4 = dataDict4['data']
    data5 = dataDict5['data']
    tstim1 = dataDict1['tstim']
    tstim2 = dataDict2['tstim']
    tstim3 = dataDict3['tstim']
    tstim4 = dataDict4['tstim']
    tstim5 = dataDict5['tstim']
    cond1 = dataDict1['condition']
    cond2 = dataDict2['condition']
    cond3 = dataDict3['condition']
    cond4 = dataDict4['condition']
    cond5 = dataDict5['condition']

    tstimall =[]
    allcond = []
    tstimall = tstim1.tolist() + (len(data1) + tstim2).tolist() + (len(data1)+len(data2)+tstim3).tolist() + \
               (len(data1) + len(data2) +len(data3) + tstim4).tolist() + \
               (len(data1) + len(data2) + len(data3) + len(data4) + tstim5).tolist()

    allrun = np.concatenate([data1, data2, data3, data4, data5], axis = 0)
    allcond = cond1 + cond2 + cond3 + cond4 + cond5

    # error checking after combining
    if [item for item, count in Counter(tstimall).items() if count > 1]:
        print("\nWarning: there exists duplicate(s) in stimulus index\n")
        exit()
    print("total trials:", len(tstimall))
    print("number of trials per cond:", len(cond1),len(cond2),len(cond3), len(cond4))

    return allrun, tstimall, allcond

def get_ica(subID, path, runica=False):
    """this function either run the ica or retrieve the ica ran before
    returns a dictionry saved under /home/jenny/ostwald-data/clean-eeg-converted/ICA/"""

    sr = 500
    print('curent subject: '+ subID)
    data, alltstim, allcond = combine_runs(subID, path)
    data = data.astype(float)

    # seperate eeg and eog ecg
    eog = data[:,30]
    ecg = data[:,31]
    badchans = [30,31]
    eeg = np.delete(data, slice(30, 32), axis=1)

    # check the data by looking at the correlation coeeficient between channels
    # fig,ax = plt.subplots()
    # im = ax.imshow(np.corrcoef(np.transpose(eeg)))
    # ax.set_title(f"Correlation Coefficients of Channels {subID}")

    # runfastica

    if runica is True:
        print("icarunning")
        ica = FastICA(n_components = 62,  whiten=True, fun = 'cube', max_iter = 10000000)
        print("running fastICA......")
        S = ica. fit_transform(eeg)
        A = ica.mixing_
        W = ica.components_

    if runica is False:
        print("retrieving ICA variables......")
        filedict = read_mat(path + 'ICA/' + subID + '_' + "ICA" + '.mat')
        S = filedict['Sources']
        A = filedict['Mixing']
        W = filedict['Components']

    # try to recover the signal
    recover = A@np.transpose(S)

    #pick a time window to see if it recovers
    fig,ax = plt.subplots(3,1)
    ax[0].plot(eeg[2000:6000,:])
    ax[0].set_title("original signal")
    ax[1].plot(S[2000:6000,:])
    ax[1].set_title("source signal")
    ax[2].plot(np.transpose(recover)[2000:6000,:])
    ax[2].set_title("recover signal")
    fig.set_size_inches(11.56,8.91)

    #
    # # baseline correction for component ERP
    # for i in range(0, trialnum):
    #     baseline_mean = np.tile(np.mean(trialcomponent[(500 - 50):500, :, i], axis=0), [trialcomponent.shape[0], 1])
    #     trialcomponent[:, :, i] = trialcomponent[:, :, i] - baseline_mean
    #
    # # plot the raw erp
    # trialeeg = np.zeros((samples, channelnum, trialnum))
    #
    # # epoch the data to create single-trial segments for 5s
    # for i in np.arange(trialnum):
    #     time = alltstim[i]
    #     trialeeg[:, :, i] = eeg[time - 1000: time + 1500, :]
    #
    # # baseline correction for raw ERP
    # for i in range(0, trialnum):
    #     baseline_mean = np.tile(np.mean(trialeeg[(500 - 50):500, :, i], axis=0), [trialeeg.shape[0], 1])
    #     trialeeg[:, :, i] = trialeeg[:, :, i] - baseline_mean
    #
    # # compare the component ERP and the eeg ERP
    # fig,ax = plt.subplots(2,1)
    # ax[0].plot(np.arange(-100,2000,2), np.mean(trialcomponent[450:1500], axis = 2))
    # ax[0].axvline(1000)
    # ax[0].axvline(0)
    # ax[0].set_title("erp in the component space")
    #
    # ax[1].plot(np.arange(-100,2000,2), np.mean(trialeeg[450:1500], axis = 2))
    # ax[1].set_title("erp from the raw data")
    # ax[1].axvline(1000)
    # ax[1].axvline(0)

    # fig.suptitle(f'{subID}')
    # fig.set_size_inches(10, 7)

    # plt.savefig((f'/home/jenny/ostwald-data/clean-eeg-converted/ICA/Figures/{subID}componentERP.png'), \
    #             dpi=300, format='png',bbox_inches='tight')


    # # get the correlation coefficient between EOG and channel from Source Matrix
    # chanmatrix = np.column_stack((S, eog))
    # R = np.corrcoef(np.transpose(chanmatrix))
    # corr =  R[-1,:]
    #
    # # find the two highest correlation channels
    # corr1 = np.argmax(abs(corr[:-1]))
    # corr2_val = sorted(abs(corr[:-1]))[-2]
    # corr2 = [x for x, i in enumerate(corr[:-1]) if i== corr2_val or i== -1 *corr2_val]
    # corr2 = int(corr2[0])
    #
    # # compare the raw eog with the highly correcltaed copmonents
    # # normalize the components
    # componentmax = max(max(S[0:10000,corr1]), max(S[0:10000,corr2]))
    # eogmin = min(eog[0:10000])
    # eogmax = max(eog[0:10000])
    # eogrange = eogmax-eogmin
    # neweog = ((eog[0:10000] - eogmin) / eogrange) * componentmax
    #
    # # epoch eog data
    # samples = int(5 * sr)
    # trialnum = len(alltstim)
    # trialeog = np.zeros((samples, trialnum))
    #
    # # epoch the data to create single-trial segments for 5s
    # for i in np.arange(trialnum):
    #     time = alltstim[i]
    #     trialeog[:, i] = eog[time - 1000: time + 1500]
    #
    # # compare the highly correlated components with and the eog channel
    #
    # componentmax = max(max(np.mean(trialcomponent[:,corr1,:], axis = 1)), max(np.mean(trialcomponent[:,corr2,:], axis = 1)))
    # eogmean = np.mean(trialeog, axis = 1)
    # eogmin = min(eogmean)
    # eogmax = max(eogmean)
    # eogrange = eogmax-eogmin
    # neweogmean = ((eogmean - eogmin) / eogrange) * componentmax


    # fig, ax = plt.subplots(3,1)
    # ax[0].scatter(np.arange(0, 62), corr[:-1])
    # ax[0].set_xlabel('channel')
    # ax[0].set_ylabel('correlation coefficient with EOG channel')
    # ax[0].plot(np.arange(0, 62), [0] * 62, 'k--')
    #
    # ax[1].plot(np.arange(0,20000,2),S[10000:20000,corr1],label = "highly correlated IC1")
    # ax[1].plot(np.arange(0,20000,2),S[10000:20000,corr2],label = "highly correlated IC2" )
    # ax[1].plot(np.arange(0,20000,2),neweog, label = "eog(re-scaled)" )
    # ax[1].legend(loc='best')
    # ax[1].set_xlabel('time(ms)')
    #
    # ax[2].plot(np.arange(-100,2000,2), np.mean(trialcomponent[450:1500,corr1,:], axis = 1),label = "correlated IC2")
    # ax[2].plot(np.arange(-100,2000,2), np.mean(trialcomponent[450:1500,corr2,:], axis = 1), label = "correlated IC2")
    # ax[2].plot(np.arange(-100,2000,2), neweogmean[450:1500], label = 'eog (re-scaled)')
    # ax[2].legend(loc='best')
    # fig.suptitle(f'{subID}')
    # fig.set_size_inches(8.7, 8.7)
    #
    # plt.savefig((f'/home/jenny/ostwald-data/clean-eeg-converted/ICA/Figures/{subID}componentEOG.png'), \
    #             dpi=300, format='png',bbox_inches='tight')

    icadict = dict()
    icadict["alltstim"] = alltstim
    icadict["allcond"] = allcond
    icadict["eeg"] = data
    # icadict["eog"] = eog
    # icadict["ecg"] = ecg
    icadict["Sources"] = S
    icadict["Mixing"] = A
    icadict["Components"] = W
    icadict["badchans"] = badchans
    icadict["sr"] = sr

    savemat(f'/home/jenny/ostwald-data/clean-eeg-converted/ICA/{subID}_ica.mat',icadict)
    return icadict

# # run through each subject
# sublist, _ = geo.get_sub(path)
# sublist = np.delete(sublist,0)
# for i in sublist:
#     get_ica(i, path, runica=False)
#     print(f"finish running {i}")
