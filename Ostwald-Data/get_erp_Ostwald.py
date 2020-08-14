from pymatreader import read_mat
import numpy as np
import timeop
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.io import savemat
import timeop
import diffusion
import os
import numpy.matlib

subID = 'sub-01'
run = '01'

path = '/home/jenny/ostwald-data/clean-eeg-converted'


# this function loads EEG data from each run
def  get_erp(subID, run):
    currentSub = subID
    currentRun = 'run-'+ run
    print('Current Subject: ', currentSub)
    print('Current Run:', currentRun)
    datadict = read_mat(path + subID + '/EEG/' + 'EEG_data_'+ subID + '_' + currentRun + '.mat')
    eventsdict = read_mat(path + subID + '/EEG/' + 'EEG_events_' + subID + '_' + currentRun + '.mat')

    data = np.array(datadict['EEGdata']['Y'])
    sr = np.array(datadict['fs'])
    tresp = np.array(eventsdict['tresp'])
    tstim = np.array(eventsdict['tstim'])

    # construct a time x channel x trial matrix for each run
    channelnum = data.shape[0]
    trialnum = tresp.shape[0]
    trialdata = np.zeros((5000,channelnum, trialnum))
    data = np.transpose(data)
    for i in np.arange(trialnum):
        time = tstim[i]
        trialdata[:,:, i] = data[time-2000: time+3000,:]
    erp = np.mean(trialdata[:,:,:],axis = 2)

    # remove the mean
    epochlength = trialdata.shape[0]
    erpmean = np.tile(np.mean(erp, axis=0), [epochlength, 1])
    erp = erp - erpmean