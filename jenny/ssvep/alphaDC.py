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
from readchans import *

path = '/home/jenny/pdmattention/'
subIDs = choose_subs(1, path + 'task3')
subIDs.remove('s236_ses1_')

# subIDs.remove('s198_ses1_')
# subIDs.remove('s218_ses2_')
# subIDs.remove('s210_ses1_')

def freqbands():
    '''this function defines the frequencies averaged for each frequency band'''
    theta = [3,4,5]
    lowalpha = [7,8,9]
    highalpha = [11,12,13]
    lowtheta = [15,16,17,18,19,20,21]
    hightheta = [23,24,25,26,27,28,29]
    bandnames = ['theta', 'lowalpha', 'highalpha', 'lowtheta','hightheta']
    bandlist = [theta, lowalpha, highalpha, lowtheta,hightheta]
    return bandlist, bandnames

def alphadc(subID, allchan=True, channame=None):
    '''this function baseline correct each electrode each trial
    allchan = True, channame = None means it's using all good chans
    allchan = False, channame = 'out.lpf' means using left prefrontal electrodes'''
    path = '/home/jenny/pdmattention/'
    _, bl_theta, bl_alphaLow, bl_alphaHigh, bl_betaLow, bl_betaHigh = baselinefreq(subID, 0, 250)
    spec = loadmat(path + 'task3/'+'spectrogram/' + subID + 'spectrogram.mat')
    frequencies=spec['frequencies'].astype(int)
    condition = spec['condition']
    correct = spec['correct']
    if allchan is True:
        goodchans = spec['goodchannels']
    if allchan is False:
        goodchannels = spec['goodchannels']
        chandict = getchans()
        groupchans = chandict[channame]
        goodchans = [i for i in groupchans if i in goodchannels]
        print("%s:" % channame, "%s" % goodchans)
    print("goodchans: %d" % len(goodchans))
    goodtrials = spec['goodtrials']
    rt = spec['rt']
    sgram = spec['spectrogram']
    nt = sgram.shape[1]
    ncond = len(np.unique(condition))
    nchan = len(goodchans)
    bandlist, bandnames = freqbands()
    nband = len(bandlist)
    freqind = []
    for l in bandlist:
        ind = [i for i, j in enumerate(frequencies) if j in l]
        freqind.append(ind)

    baseline = bl_theta, bl_alphaLow, bl_alphaHigh, \
                bl_betaLow, bl_betaHigh
    baseline = np.array(baseline)

    # baseline correction
    alphaDC = np.zeros((nband, nt, 129, 360))
    for t in range(0,360):
        for c in range(0,128):
            #this is the power of each electrode of each trial
            chansgram = sgram[:,:,c,t]
            chanbl = baseline[:,c,t]
            for i in range(nband):
                relativepower = np.abs(chansgram[freqind[i],:])**2 / chanbl[i]
                relativepower = np.median(relativepower, axis=0) # this is per electrode per trial per frequency band
                alphaDC[i,:,c,t] = relativepower
    # np.save(path +'/alphaDC' + '/desynch_%s'% subID[0:5], alphaDC)
    alpha2 = alphaDC[:,:,goodchans,:]

    alpha3 = alpha2[:,:,:,goodtrials]

    # by condition
    conditionpower = np.zeros((nt, ncond, nband))
    for k in range(ncond):
        trials = np.intersect1d(np.where(condition == k+1)[0],goodtrials)
        alpha = np.median(alpha2[:,:,:,trials], axis=(2,3))
        conditionpower[:,k,:] = alpha.T

    # by RT
    rtpower = np.zeros((nt, ncond, nband))
    for k in range(3):
        npercond = np.floor(len(goodtrials)/3)
        trials = [j for i, j in enumerate(goodtrials) if i in np.arange(npercond*k,npercond*k+npercond,1)]
        alpha = np.median(alpha2[:,:,:,trials], axis=(2,3))
        rtpower[:, k, :] = alpha.T

    # by accuracy
    accpower = np.zeros((nt,2,nband))
    for k in range(0,2):
        trials = np.intersect1d(np.where(correct == k)[0], goodtrials)
        alpha = np.median(alpha2[:,:,:,trials], axis=(2,3))
        accpower[:,k,:] = alpha.T

    # by four condition
    accrtpower = np.zeros((nt,4,nband))
    nhalf = np.floor(len(goodtrials)/2)
    fasttrials = goodtrials[0:int(nhalf)]
    slowtrials = goodtrials[int(nhalf):]
    for k in range(0,2):
        trials = np.intersect1d(np.where(correct == k)[0], fasttrials)
        alpha = np.median(alpha2[:,:,:,trials], axis=(2,3))
        accrtpower[:,k,:] = alpha.T
    for k in range(0, 2):
        trials = np.intersect1d(np.where(correct == k)[0], slowtrials)
        alpha = np.median(alpha2[:, :, :, trials], axis=(2, 3))
        accrtpower[:, k+2, :] = alpha.T

    return alphaDC, conditionpower,rtpower, accpower, accrtpower

    # return conditionpower, rtpower

def baselinefreq(subID, tstart,tend):
    '''this function returns raw fft coeficient for the baseline, and the power at 4Hz, 8Hz, 12Hz, 16&20, 24&28 at each channel for each trial
    tstart is where the baseline starts'''
    path = '/home/jenny/pdmattention/task3/'
    currentSub = subID[0:4]
    print('Current Subject: ', currentSub)
    datadict = read_mat(path + subID + 'task3_final.mat')
    data = np.array(datadict['data'])
    sr = np.array(datadict['sr'])
    baseline = data[tstart:tend,:,:]
    n = baseline.shape[0]
    ntrials = baseline.shape[2]
    nchans = baseline.shape[1]

    hann = np.hanning(n)
    newbaseline = np.zeros((n, baseline.shape[1],ntrials))
    for i in range(0,ntrials):
        hann2d = np.tile(hann, (129,1)).T
        newbaseline[:,:,i] = baseline[:,:,i]*hann2d
    basef = fft(newbaseline, axis=0)
    f = 1000 * np.arange(0, n / 2) / n
    bl_theta, bl_alphaLow, bl_alphaHigh, bl_betaLow, bl_betaHigh = \
        np.abs(basef[1,:,:]/n)**2,np.abs(basef[2,:,:]/n)**2,np.abs(basef[3,:,:]/n)**2, \
        np.mean(np.abs(basef[4:6,:,:]/n)**2, axis=0), np.mean(np.abs(basef[6:8,:,:]/n)**2, axis=0)
    return basef, bl_theta, bl_alphaLow, bl_alphaHigh, bl_betaLow, bl_betaHigh

###################################### plots ################################333
timev = np.arange(-750,1250,10)
# plots by condition
def plotcond(data_sub):
    if len(data_sub.shape) > 3:
        subjdata = np.mean(data_sub, axis=0)
    else:
        subjdata = data_sub
    fig, ax = plt.subplots(1,subjdata.shape[2],figsize=(24,5))
    label = ['easy','medium','hard']
    title = ['theta(3-5Hz)', 'low alpha(7-9Hz)', 'high alpha(11-13Hz)', 'low beta(15-21Hz)', 'high beta(23-29Hz)']
    # label = ['fast','medium','slow']
    for j in range(0,5):
        for i in range(0, 3):
            x = subjdata[50:250, i, j]
        # x = x - np.mean(x)
        # x = x - np.mean(x[100:125])
            ax[j].plot(timev,x, label=label[i])
            ax[j].axvline(0,ls='--',color='grey')
            ax[j].axhline(0, ls='--', color='grey')
            ax[j].legend()
            ax[j].set_title(title[j])
            ax[j].set_xlabel('Time(ms)')
            ax[j].xaxis.set_ticks(np.arange(-750, 1250, 250))
    fig.suptitle('Stimulus-Locked Desynchronization by Condition (')
    fig.savefig(path + 'alphaDC/plots/bycond_test')
    # plt.close(fig)


# plots by RT
def plotrt(data_sub):
    if len(data_sub.shape) > 3:
        subjdata = np.mean(data_sub, axis=0)
    else:
        subjdata = data_sub
    fig, ax = plt.subplots(1,subjdata.shape[2],figsize=(24,5))
    label = ['fast','mid','slow']
    title = ['theta(3-5Hz)', 'low alpha(7-9Hz)', 'high alpha(11-13Hz)', 'low beta(15-21Hz)', 'high beta(23-29Hz)']
    # label = ['fast','medium','slow']
    for j in range(0,5):
        for i in range(0, 3):
            x = subjdata[50:250, i, j]
        # x = x - np.mean(x)
        # x = x - np.mean(x[100:125])
            ax[j].plot(timev,x, label=label[i])
            ax[j].axvline(0,ls='--',color='grey')
            ax[j].axhline(0, ls='--', color='grey')
            ax[j].legend()
            ax[j].set_title(title[j])
            ax[j].set_xlabel('Time(ms)')
            ax[j].xaxis.set_ticks(np.arange(-750, 1250, 250))
    fig.suptitle('Stimulus-Locked Desynchronization by RT')
    fig.savefig(path + 'alphaDC/plots/byrt_test')
    # plt.close(fig)

# plots by correct
def plotacc(data_sub):
    if len(data_sub.shape) > 3:
        subjdata = np.mean(data_sub, axis=0)
    else:
        subjdata = data_sub
    fig, ax = plt.subplots(1, subjdata.shape[2],figsize=(24,5))
    label = ['incorrect','correct']
    title = ['theta(3-5Hz)', 'low alpha(7-9Hz)', 'high alpha(11-13Hz)', 'low beta(15-21Hz)', 'high beta(23-29Hz)']
        # label = ['fast','medium','slow']
    for j in range(0, 5):
        for i in range(0, 2):
            x = subjdata[50:250, i, j]
            ax[j].plot(timev, x, label=label[i])
            ax[j].axvline(0, ls='--', color='grey')
            ax[j].axhline(0, ls='--', color='grey')
            ax[j].legend()
            ax[j].set_title(title[j])
            ax[j].set_xlabel('Time(ms)')
            ax[j].xaxis.set_ticks(np.arange(-750, 1250, 250))
    fig.suptitle('Stimulus-Locked Desynchronization by Accuracy')
    fig.savefig(path + 'alphaDC/plots/byacc_test')
    # plt.close(fig)

def plotaccrt(data_sub):
    if len(data_sub.shape) > 3:
        subjdata = np.mean(data_sub, axis=0)
    else:
        subjdata = data_sub
    fig, ax = plt.subplots(1, subjdata.shape[2],figsize=(24,5))
    label = ['fast incorrect', 'fast correct', 'slow incorrect','slow correct']
    title = ['theta(3-5Hz)', 'low alpha(7-9Hz)', 'high alpha(11-13Hz)', 'low beta(15-21Hz)', 'high beta(23-29Hz)']
    for j in range(0, 5):
        for i in range(0, 4):
            x = subjdata[50:250, i, j]
            ax[j].plot(timev, x, label=label[i])
            ax[j].axvline(0, ls='--', color='grey')
            ax[j].axhline(0, ls='--', color='grey')
            ax[j].legend()
            ax[j].set_title(title[j])
            ax[j].set_xlabel('Time(ms)')
            ax[j].xaxis.set_ticks(np.arange(-750, 1250, 250))
    fig.suptitle('Stimulus-Locked Desynchronization by Accuracy and RT')
    fig.savefig(path + 'alphaDC/plots/byaccrt_test')
    # plt.close(fig)


############################################################################

# get all subjects
nband = 5
subject_bycond= np.zeros((len(subIDs),400,3,nband))
subject_rt = np.zeros((len(subIDs),400,3,nband))
subject_acc = np.zeros((len(subIDs),400,2,nband))
subject_accrt = np.zeros((len(subIDs),400,4,nband))

count = 0
for sub in subIDs:
    _, conditionpower,rtpower, accpower, accrtpower = alphadc(sub, allchan = True, channame = None)
    subject_bycond[count,:,:,:]=conditionpower
    subject_rt[count,:,:,:]=rtpower
    subject_acc[count,:,:,:]=accpower
    subject_accrt[count,:,:,:]=accrtpower
    count +=1
    print('%d' %  count, '/', '%d'%len(subIDs))

np.save(path + 'alphaDC/'+'/subject_bycond_lf', subject_bycond)
np.save(path + 'alphaDC/'+'/subject_rt_lf', subject_rt)
np.save(path + 'alphaDC/'+'/subject_acc_lf', subject_acc)
np.save(path + 'alphaDC/'+'/subject_accrt_lf', subject_accrt)

subject_bycond = np.load(path + 'alphaDC/'+'subject_bycond_all.npy')
subject_rt = np.load(path + 'alphaDC/'+'subject_rt_all.npy')
subject_acc = np.load(path + 'alphaDC/'+'subject_acc_all.npy')
subject_accrt = np.load(path + 'alphaDC/'+'subject_accrt_all.npy')

plotcond(subject_bycond)
plotrt(subject_rt)
plotacc((subject_acc))
plotaccrt(subject_accrt)

