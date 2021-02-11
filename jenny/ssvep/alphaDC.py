#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 16:05:45 2020

@author: jenny
"""

# this scripts contain functions for alpha DC analysis

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
from transfertools.models import TCA, CORAL
#%%
# import lab modules
import timeop
from ssvep_subject import *

subIDs = choose_subs(1, path + 'task3')
subIDs.remove('s236_ses1_')

def baselinefreq(subID):
    path = '/home/ramesh/pdmattention/task3/'
    currentSub = subID[0:4]
    print('Current Subject: ', currentSub)
    pcdict = read_mat(path + subID + 'task3_photocells.mat')
    datadict = read_mat(path + subID + 'task3_final.mat')
    behavdict = read_mat(path + subID[0:5] + 'behavior_final.mat')

    data = np.array(datadict['data'])
    artifact = np.array(datadict['artifact'])
    sr = np.array(datadict['sr'])
    beh_ind = np.array(behavdict['trials'])
    condition = np.array(behavdict['condition'])
    rt = behavdict['rt']
    correct = behavdict['correct']

    baseline = data[0:250,]

    # %%
    debug = True
    lvlAnalysis = 1
    path = '/home/ramesh/pdmattention/task3/spectrogram/'
    subIDs = choose_subs(lvlAnalysis, path)
    ns = len(subIDs)
    motorchannels = [36, 30, 29, 35, 41, 42, 37]
    # motorchannels = [104,110,111,105,87,93,103]
    motorchannels = [72, 62, 67, 71, 75, 76, 77]
    subjallpower = np.zeros((28, 400, ns))
    subjcondpower = np.zeros((28, 400, 3, ns))
    subjrtpower = np.zeros((28, 400, 3, ns))
    meanrtbycond = np.zeros((3, ns))
    medianrtbycond = np.zeros((3, ns))
    meanrt = np.zeros((3, ns))
    medianrt = np.zeros((3, ns))
    scount = 0
    x = loadmat('/home/ramesh/MNI_HEADMODEL/HNL128_Laplacian.mat')
    lap = x['lap']
    lapchan = x['goodchan'][0]
    for subID in subIDs:
        print(subID)
        # extract variables needed
        data = loadmat(path + subID + '_spectrogram.mat')
        sgram = data['spectrogram']
        goodtrials = data['goodtrials']
        condition = data['condition']
        rt = data['rt']
        frequencies = data['frequencies']
        correct = data['correct']
        stimtime = data['stimulustime']
        time = data['time']
        #   compute surface Laplacian
        sgramlap = np.zeros((28, 400, 128, 360)) + 1j * np.zeros((28, 400, 128, 360))
        for j in range(28):
            for k in range(360):
                fdata = np.squeeze(sgram[j, :, :, k])
                fdata2 = fdata[:, lapchan.astype(int)] @ np.transpose(lap)
                sgramlap[j, :, :, k] = fdata2
        # compute overall power measures
        allpower = np.abs(sgram[:, :, :, goodtrials])
        relpower = relativepower(allpower[:, :, np.arange(0, 128, 1), :], np.arange(75, 125, 1))


