#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 12:30:08 2020

@author: ramesh
"""


# import python modules
from pymatreader import read_mat
import numpy as np
import scipy.signal as signal
from scipy import linalg
from scipy.io import savemat
from matplotlib import pyplot as plt
from scipy.fftpack import fft


#%%
# import lab modules
import timeop
import diffusion
#%%

def getdata(path,subID):
    currentSub = subID[0:4]
    print('Current Subject: ', currentSub)
#    pcdict = read_mat(path + subID + '_task3_photocells.mat')
    datadict = read_mat(path + subID + '_task3_final.mat')
    behavdict = read_mat(path + subID[0:4] + '_behavior_final.mat')
    
    data = np.array(datadict['data'])

    artifact = np.array(datadict['artifact'])
    sr = np.array(datadict['sr'])
    beh_ind = np.array(behavdict['trials'])

    # open up indices
    artifact0 = artifact.sum(axis=0)
    artifact1 = artifact.sum(axis=1)

    # identify goodtrials and good channels.
    goodtrials = np.squeeze(np.array(np.where(artifact0 < 20)))
    goodchans = np.squeeze(np.array(np.where(artifact1 < 40)))

    # BehEEG_int = list(set(beh_ind) & set(goodtrials))
    finalgoodtrials = np.array(diffusion.compLists(beh_ind, goodtrials))
    # finalgoodtrials = np.array(BehEEG_int)
    return data,sr,finalgoodtrials,goodchans
#%%
#globals 
path = '/home/ramesh/pdmattention/task3/'
subID='s195_ses2'
data,sr,goodtrials,goodchans = getdata(path,subID)
sos,w,h = timeop.makefiltersos(sr,15,20)  #lowpass filter at 15 hz, needs high pass too which depends on window - discuss.  
sos_hi,w_h1,h_h1 = timeop.makefiltersos(sr,5,2.5)
trialnumber = goodtrials[10]
trialdata = np.squeeze(data[:,:,trialnumber])
trialdatafilt = signal.sosfiltfilt(sos_hi,trialdata,axis = 0,padtype='odd')
trialdatafilt = signal.sosfiltfilt(sos,trialdatafilt,axis = 0,padtype='odd')
#resample at 100 Hz. factor of 10
time = np.linspace(0,3990,400) 
trialdataresample =  trialdatafilt[time.astype(int),:] 

