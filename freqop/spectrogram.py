#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:33:21 2020

@author: ramesh
"""
#%%
import numpy as np
import pywt
from scipy.signal import savgol_filter
#%% go from frequency to scale. 
def freq2scale(frequencies,fs):
    ''' Assumes that normalized center frequency is 1 '''
    dt = 1/fs
    scales = 1/frequencies
    scales = scales/dt
    return scales
#%%
def waveletspecgram(data,sr,frequencies,goodchannels,goodtrials,waveletname,decimate):
    scales = freq2scale(frequencies,sr)
    time = np.arange(decimate-1,data.shape[0],decimate)
    specgram = np.zeros((len(frequencies),len(time),data.shape[1],data.shape[2]))+1j*np.zeros((len(frequencies),len(time),data.shape[1],data.shape[2]))
    dt = 1/sr
    sg = np.floor(1/(2*dt))*2+1
    for k  in goodtrials:
    	print(k)
    	trialdata = np.squeeze(data[:,goodchannels,k])
    	trialspecgram,freqs = pywt.cwt(trialdata,scales,waveletname,sampling_period=dt,method='fft',axis=0)
    	for l in range(len(frequencies)):
    		sg = np.floor(1/(frequencies[l]*dt))*2+1
    		ts1 = savgol_filter(np.squeeze(np.real(trialspecgram[l,:,:])),int(sg),1,axis = 0,mode = 'interp')
    		ts2 = savgol_filter(np.squeeze(np.imag(trialspecgram[l,:,:])),int(sg),1,axis = 0,mode = 'interp')
    		ts3 = ts1+1j*ts2
    		specgram[l,:,goodchannels,k] = np.transpose(ts3[time,:])       
    return specgram, freqs, time
#%%

