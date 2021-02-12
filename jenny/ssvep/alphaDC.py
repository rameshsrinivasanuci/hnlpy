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



subject_bycond= np.zeros((len(subIDs),400,3,4))
subject_rt = np.zeros((len(subIDs),400,3,4))

count = 0
for sub in subIDs:
    conditionpower, rtpower=alphadc(sub)
    subject_bycond[count,:,:,:]=conditionpower
    subject_rt[count,:,:,:]=rtpower
    count +=1



def alphadc(subID):
    '''this code get '''
    basef, bl_low, bl_alphalow, bl_alphaligh,bl_high = baselinefreq(subID)
    spec = loadmat(path + 'task3/'+'spectrogram/' + subID + 'spectrogram.mat')
    frequencies=spec['frequencies']
    condition = spec['condition']
    correct = spec['correct']
    goodchans = spec['goodchannels']
    goodtrials = spec['goodtrials']
    rt = spec['rt']
    sgram = spec['spectrogram']
    ind1, ind2, ind3, ind4 = frequencies.tolist().index(2), frequencies.tolist().index(6), frequencies.tolist().index(10), frequencies.tolist().index(20)
    ind = np.arange(ind1,ind1+3), np.arange(ind2,ind2+5),np.arange(ind3, ind3+4), np.arange(ind4, ind4+10)
    nt = sgram.shape[1]

    conditionpower = np.zeros((nt,3,4))
    baseline = bl_low[goodchans,:], bl_alphalow[goodchans,:], bl_alphaligh[goodchans,:], bl_high[goodchans,:]
    for k in range(3):
        trials = np.intersect1d(np.where(condition == k+1)[0],goodtrials)
        for band in range(0,4):
            condpower =np.mean(np.abs(sgram[:, :, :, trials]**2), axis=3)
            condpower = np.mean(condpower[ind[band].tolist(),:], axis=0)
            condpower = np.mean(condpower, axis=1)
            bl = np.mean(baseline[band][:,trials], axis=(0,1))
            conditionpower[:,k,band]=condpower/bl

    rtpower = np.zeros((nt,3,4))
    for k in range(3):
        trials = [i for i in goodtrials if i in np.arange(120*k,120*k+120,1)]
        for band in range(0,4):
            rtp =np.mean(np.abs(sgram[:, :, :, trials]**2), axis=3)
            rtp = np.mean(rtp[ind[band].tolist(),:], axis=0)
            rtp = np.mean(rtp, axis=1)
            bl = np.mean(baseline[band][:,trials], axis=(0,1))
            rtpower[:,k,band]=rtp/bl
    return conditionpower, rtpower




def baselinefreq(subID):
    '''this function returns raw fft coeficient for the baseline, and the power at 4Hz, 8Hz and 12Hz at each channel for each trial'''
    path = '/home/jenny/pdmattention/task3/'
    currentSub = subID[0:4]
    print('Current Subject: ', currentSub)
    datadict = read_mat(path + subID + 'task3_final.mat')

    data = np.array(datadict['data'])
    artifact = np.array(datadict['artifact'])
    sr = np.array(datadict['sr'])

    baseline = data[0:250,:,:]
    n = baseline.shape[0]
    ntrials = baseline.shape[2]
    hann = np.hanning(n)
    newbaseline = np.zeros((n, baseline.shape[1],ntrials))
    for i in range(0,ntrials):
        hann2d = np.tile(hann, (129,1)).T
        newbaseline[:,:,i] = baseline[:,:,i]*hann2d
    basef = fft(baseline, axis=0)
    f = 1000 * np.arange(0, n / 2) / n
    bl_low, bl_alphaLow, bl_alphaHigh, bl_high = np.abs(basef[1,:,:]/n)**2,np.abs(basef[2,:,:]/n)**2,np.abs(basef[3,:,:]/n)**2, np.mean(np.abs(basef[5:7,:,:]/n)**2, axis=0)
    return basef, bl_low, bl_alphaLow, bl_alphaHigh, bl_high




np.save(path+'/alphaDC', subject_bycond)
np.save(path+'/alphaDC', subject_rt)

def makeplots():
    conditionpower  = np.mean(subject_bycond, axis=0)
     label = ['easy','medium','hard']
    # label = ['fast','medium','slow']

    timev = np.arange(-1150,1500,10)
    fig, ax = plt.subplots(1,3)

    for i in range(0, 3):
        x = conditionpower[10:275, i, 1]
        x = x - np.mean(x)
        # x = x - np.mean(x[100:125])
        ax[0].plot(timev,x, label=label[i])
        ax[0].legend()
        ax[0].axvline(0,ls='--',color='grey')
        ax[0].axhline(0, ls='--', color='grey')
        ax[0].set_title('low alpha (6-10Hz)')
        ax[0].set_xlabel('Time(ms)')



    for i in range(0, 3):
        x = conditionpower[10:275, i, 2]
        x = x - np.mean(x)
        # x = x - np.mean(x[100:125])
        ax[1].plot(timev,x, label=label[i])
        ax[1].legend()
        ax[1].axvline(0,ls='--',color='grey')
        ax[1].axhline(0, ls='--', color='grey')
        ax[1].set_title('high alpha (11-14Hz)')
        ax[1].set_xlabel('Time(ms)')


    for i in range(0, 3):
        x = conditionpower[10:275, i, 0]
        x = x - np.mean(x)
        # x = x - np.mean(x[100:125])
        ax[2].plot(timev,x, label=label[i])
        ax[2].legend()
        ax[2].axvline(0,ls='--',color='grey')
        ax[2].axhline(0, ls='--', color='grey')
        ax[2].set_title('low band (2-4Hz)')
        ax[2].set_xlabel('Time(ms)')

    fig.suptitle('Stimulus-Locked Desynchronization by Cond (corrected)')



