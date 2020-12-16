#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 07:05:09 2020

@author: ramesh
"""

#analysis script for beta desynchronization 
#this is very preliminary
#%%
#import python modules
from pymatreader import read_mat
from hdf5storage import savemat
from hdf5storage import loadmat
import hdf5storage
import numpy as np
import pywt
from scipy.signal import savgol_filter
import os
import numpy as np
#import lab modules 
#%%
def choose_subs(lvlAnalysis, path):
	# this function returns a list of subject IDs and task based on whether or not this is testing
	# analysis or analysis of all subjects
	# returns

	allDataFiles = os.listdir(path)

	if lvlAnalysis == 1:
		excludeSubs = ['s184', 's187', 's190', 's193', 's199', 's209', 's214', 's220', 's225', 's228', \
		's234', 's240', 's213', 's235']

		subIDs = [] 

		for files in allDataFiles:
			if (files[10:] == 'spectrogram.mat') and ((files[:4] not in excludeSubs) == True):
				subIDs.append(files[0:9])

	elif lvlAnalysis == 2:
		subIDs = []
		for files in allDataFiles:
			if files[10:] == 'spectrogram.mat':
				subIDs.append(files[0:9])
 
	return subIDs

#%%
def relativepower(spectrogram,baseline):
    #this function expresses power relative to the mean power in a baseline interval given by baseline. 
    #assumes frequency x time x channel
    nf = spectrogram.shape[0]
    nt = spectrogram.shape[1]
    nc = spectrogram.shape[2]
    ntrial = spectrogram.shape[3]
    relpower = np.zeros((nf,nt,nc,ntrial))
    for j in range(nf):
        for k in range(nc):
            for l in range(ntrial):
                baselinepower = np.mean(np.squeeze(spectrogram[j,baseline,k,l]))
                relpower[j,:,k,l] = spectrogram[j,:,k,l]/baselinepower
    return relpower
    
#%%
debug = True
lvlAnalysis = 1
path = '/home/ramesh/pdmattention/task3/spectrogram/'
subIDs = choose_subs(lvlAnalysis, path)
ns = len(subIDs)
motorchannels = [36,30,29,35,41,42,37]
#motorchannels = [104,110,111,105,87,93,103]
motorchannels = [72,62,67,71,75,76,77]
subjallpower = np.zeros((28,400,ns))
subjcondpower = np.zeros((28,400,3,ns))
subjrtpower = np.zeros((28,400,3,ns))
meanrtbycond = np.zeros((3,ns))
medianrtbycond = np.zeros((3,ns))
meanrt = np.zeros((3,ns))
medianrt = np.zeros((3,ns))
scount = 0
# x = loadmat('/home/ramesh/MNI_HEADMODEL/HNL128_Laplacian.mat')
# lap = x['lap']
# lapchan = x['goodchan'][0]
for subID in subIDs:
    print(subID)
    #extract variables needed 
    data = loadmat(path + subID + '_spectrogram.mat') 
    sgram = data['spectrogram']
    goodtrials = data['goodtrials']
    condition = data['condition']
    rt = data['rt']
    frequencies = data['frequencies']
    correct = data['correct']
    stimtime = data['stimulustime']
    time = data['time'] 
# #   compute surface Laplacian
#     sgramlap= np.zeros((28,400,128,360))+1j*np.zeros((28,400,128,360))
#     for j in range(28):
#         for k in range(360):
#             fdata = np.squeeze(sgram[j,:,:,k])
#             fdata2 = fdata[:,lapchan.astype(int)]@np.transpose(lap)
#             sgramlap[j,:,:,k] = fdata2
    #compute overall power measures
    allpower = np.abs(sgram[:,:,:,goodtrials])
    relpower = relativepower(allpower[:,:,np.arange(0,128,1),:],np.arange(75,125,1))    


