#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 16:05:45 2020

@author: jenny
"""

# this scripts calculate the ssvep power for each subject

# %%
# import python modules
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
#%%

path = '/home/ramesh/pdmattention/task3/'


def trial_ssvep(subID, freq):
    '''this function returns the power and snr of single trials
    when frequency is 30 and 40 for all conditions'''
    stimulus_ssvep, noise_ssvep, photocell, _, _, _, behav_final = SSVEP_task3(subID)
    stim_estimate = stimulus_ssvep['trial_bychan']
    valid_chans = np.where(stimulus_ssvep['erp_fft'][0,] != 0)
    if len(valid_chans[0]) != 121:
        print('WARNING: %i channels with 0s detected, using the defalt 121 channels'% len(valid_chans[0]))
        valid_chans, _ = default_validchans()
    stim_Signal =  2 * np.abs(stim_estimate[freq,valid_chans,:]) **2
    stim_Noise =  2 * (1/2 * (np.abs(stim_estimate[freq-1,valid_chans,:]) **2 + np.abs(stim_estimate[freq+1,valid_chans,:]) **2))
    StimSnr = np.squeeze(np.divide(stim_Signal, stim_Noise, out=np.zeros_like(stim_Signal), where=stim_Noise != 0))
    finalgoodtrials = stimulus_ssvep['goodtrials']
    acc = behav_final['acc']
    rt = behav_final['rt']
    rt_tertile = np.percentile(rt,[33.33, 66.66])
    rt_class = np.zeros(len(rt))
    rt_class[(rt > rt_tertile[0]) & (rt <= rt_tertile[1])] = 1
    rt_class[rt > rt_tertile[1]] = 2
    return StimSnr, finalgoodtrials, acc, rt, rt_class


def subject_average(subID):
    '''this function returns the power and snr when frequency is 30 and 40 for all conditions'''
    stimulus_ssvep, noise_ssvep, photocell, behavdict, _, _, _ = SSVEP_task3(subID)
    StimSnr, StimPower = get_power(stimulus_ssvep, behavdict, 30)
    NoiseSnr, NoisePower = get_power(stimulus_ssvep, behavdict, 40)
    pc_chans = np.where(StimSnr == 0)
    print('photocell channels to skip:'+ str(pc_chans))
    return StimSnr, StimPower, NoiseSnr, NoisePower, pc_chans

def subject_bycond(subID, freq):
    '''this function returns the power and snr when frequency is 30 and 40 by condition'''
    _,_,_, behavdict, stim_erpf, noise_erpf, _ =  SSVEP_task3(subID)
    StimPower = 2 * np.abs(stim_erpf[freq,:,:]) **2
    StimNeighbourPower =  2 * (1/2 * (np.abs(stim_erpf[freq-1,:,:]) **2 + np.abs(stim_erpf[freq+1,:,:]) **2))
    StimSnr = np.divide (StimPower, StimNeighbourPower, out=np.zeros_like(StimPower), where=StimNeighbourPower!=0)

    NoisePower = 2 * np.abs(noise_erpf[freq,:,:]) **2
    NoiseNeighbourPower =  2 * (1/2 * (np.abs(noise_erpf[freq-1,:,:]) **2 + np.abs(noise_erpf[freq+1,:,:]) **2))
    NoiseSnr = np.divide (NoisePower, NoiseNeighbourPower, out=np.zeros_like(NoisePower), where=NoiseNeighbourPower!=0)
    return StimSnr, StimPower, NoiseSnr, NoisePower


def get_power(ssvep, behavdict, freq):
    '''this function returns the power and snr after fft the ERPs'''
    # this is the the 1000 * 128 fft from erp
    erpf = ssvep['erp_fft']
    goodchans = ssvep['goodchannels']
    goodtrials = ssvep['goodtrials']
    condition = behavdict['condition']

    # get the power at 30Hz (signal) and 29Hz + 31Hz (noise) from all channels
    signal_power = 2 * np.abs(erpf[freq,:]) **2
    noise_power = 2 * (1/2 * (np.abs(erpf[freq-1,:]) **2 + np.abs(erpf[freq+1,:]) **2))

    # get the snr, the division ==0 for all photocell channels
    snr = np.divide (signal_power, noise_power, out=np.zeros_like(signal_power), where=noise_power!=0)
    return snr, signal_power


    # plotting the spectra of the erp after svd
    sr = 1000
    nyquist = sr/2
    xf = np.linspace(0.0, nyquist, len(stim_erpf) // 2 +1)
    plt.plot(xf[10:40], (2 * np.abs(stim_erpf)[10:40,:]) ** 2)



def getSSVEP(data,sr,window,ssvep_freq,goodtrials,goodchans):
    """ this function generates ssvep structure including fourier coefficients of
     erp, ssvep power and output from svd procedure """

    SSVEP = dict();

    #some renaming
    startsamp = window[0]
    endsamp = window[1]
    epochlength = data.shape[0]
    nchan = data.shape[1]
    ntrial = data.shape[2]

    # average erp.
    erp = np.mean(data[:, :, goodtrials], axis=2)    

    # FFT the ERPs
    # remove the mean
    erpmean = np.tile(np.mean(erp, axis=0), [epochlength, 1])
    erp = erp - erpmean

    #take fft over prescribed window
    erpf = fft(erp[startsamp:endsamp, :], axis=0)/(endsamp-startsamp) # raw fft     
    binwidth = int((endsamp-startsamp)/sr)
    u,s,vh = linalg.svd(erpf[(ssvep_freq-1)*binwidth:(ssvep_freq+1)*binwidth+1,:])
    snr = 2 * (np.abs(u[1,:]**2))/(np.abs(u[0,:])**2 + np.abs(u[2,:])**2)

    snrflagsignal = 1
    if np.max(snr) < 1:
        print('Warning NO SSVEP detected at stimulus frequency')
        snrflagsignal = 0
    
    weights = np.zeros((nchan,1),dtype=complex)

	# This is an optimal set of weights to estimate 30 hz signal. 
    weights[:,0] = np.matrix.transpose(vh[0,:])

	# lets test it on the same interval using weighted electrode vs. original
    erpproject = np.matmul(erpf, weights)

    # multiply the weights for all timepoints
    weights_long = np.tile(weights * np.diag(s)[0,0], [1,1000])
    channel_power = np.transpose(erpf) * weights_long

    # Now use the weights to loop through indivial trials 
    trialestimate = np.zeros((endsamp-startsamp,ntrial),dtype=complex)
    trial_bychan = np.zeros((endsamp-startsamp,nchan, ntrial), dtype = complex)
    trial_fft = np.zeros((endsamp-startsamp,nchan, ntrial), dtype = complex)
    trial_data = np.zeros((endsamp-startsamp,nchan, ntrial))

    for trial in goodtrials: 
        trialdata = np.squeeze(data[startsamp:endsamp,:,trial])
        trialfft = fft(trialdata,axis=0)
        trialproject = np.matmul(trialfft,weights_long)
        trialfft_weighted = trialfft * weights_long.T
        trial_bychan[:,:,trial] = trialfft_weighted
        trial_fft[:, :, trial] = trialfft
        trialestimate[:,trial] = trialproject[:,0] #new coefficients
        trial_data[:,:,trial] = trialdata


    SSVEP['goodtrials'] = goodtrials
    SSVEP['goodchannels'] = goodchans
    SSVEP['sr'] = sr;
    SSVEP['ssvep_freq'] = ssvep_freq
    SSVEP['samplerange'] = window
    SSVEP['erp_fft'] = erpf
    SSVEP['svdspectrum'] = u
    SSVEP['svdchan'] = vh[0:2,:]
    SSVEP['snr'] = snr
    SSVEP['snrflag'] = snrflagsignal
    SSVEP['projectspectrum'] = erpproject
    SSVEP['singletrial'] = trialestimate
    SSVEP['weights'] = weights
    SSVEP['power_sub'] = channel_power
    SSVEP['singular'] = np.diag(s)
    SSVEP['trial_bychan'] = trial_bychan
    SSVEP['trialfft'] = trial_fft
    SSVEP['trialdata'] = trial_data
    return SSVEP
    

#%%
# globals
def SSVEP_task3(subID):
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

    # open up indices
    artifact0 = artifact.sum(axis=0)
    artifact1 = artifact.sum(axis=1)

    # identify goodtrials and good channels.
    goodtrials = np.squeeze(np.array(np.where(artifact0 < 20)))
    goodchans = np.squeeze(np.array(np.where(artifact1 < 40)))

    # choosing the trials that have RT over 300ms and check if they are all goodtrials
    # get the index and apply to rt, condition and accuracy
    xy, x_ind, y_ind = np.intersect1d(beh_ind, goodtrials, return_indices=True)
    ind, finalgoodtrials = np.array(compListsInd(beh_ind, goodtrials))
    # here the ind stands for ind in beh_ind that are qualified as goodtrials
    # finalgoodtrials stand for the actual trial index from the original unsorted dataset
    rt_final = rt[ind]
    acc_final = correct[ind]
    behav_final = {'rt': rt_final, 'acc': acc_final}
    condition_final = condition[ind]
    correct_final = correct[ind]

    # get the photocells
    photocell = dict()
    p = pcdict['photostim']
    n = pcdict['photonoise']
    yfp = fft(p[1250:2250,:], axis=0)
    yfn = fft(n[1250:2250,:], axis=0)
    photocell['stim'] = p
    photocell['noise'] = n
    photocell['stim_fft'] = yfp
    photocell['noise_fft'] = yfn

    # time window of interest
    window = [1250, 2250]

    # FFT the eeg data
    stimulus_ssvep = getSSVEP(data,sr,window,30,finalgoodtrials,goodchans)
    noise_ssvep = getSSVEP(data,sr,window,40,finalgoodtrials,goodchans)

    # same procedure analyzed by condition
    ind_ez = np.where(condition_final == 1)
    ind_md = np.where(condition_final == 2)
    ind_hr = np.where(condition_final == 3 )

    stim_ez = getSSVEP(data,sr,window,30, finalgoodtrials[ind_ez], goodchans)
    stim_md = getSSVEP(data, sr, window, 30, finalgoodtrials[ind_md], goodchans)
    stim_hr = getSSVEP(data, sr, window, 30, finalgoodtrials[ind_hr], goodchans)
    noise_ez = getSSVEP(data,sr,window, 40, finalgoodtrials[ind_ez], goodchans)
    noise_md = getSSVEP(data, sr, window, 40, finalgoodtrials[ind_md], goodchans)
    noise_hr = getSSVEP(data, sr, window, 40, finalgoodtrials[ind_hr], goodchans)

    stim_erpf = np.dstack((stim_ez['erp_fft'], stim_md['erp_fft'], stim_hr['erp_fft']))
    noise_erpf = np.dstack((noise_ez['erp_fft'], noise_md['erp_fft'], noise_hr['erp_fft']))

    return stimulus_ssvep, noise_ssvep, photocell, behavdict, stim_erpf, noise_erpf, behav_final


def default_validchans():
    '''this function returns the channels that are non photocells and photocells'''
    stimulus_ssvep, _, _, _, _, _,_ = SSVEP_task3('s239_ses1_')
    valid_chans = np.where(stimulus_ssvep['erp_fft'][0,] != 0)
    pc_chans = np.where(stimulus_ssvep['erp_fft'][0,] == 0)
    return valid_chans, pc_chans

# here are some reivsed functions from diffusion.py by Mariel

def choose_subs(lvlAnalysis, path):

    allDataFiles = os.listdir(path)

    if lvlAnalysis == 1:
        excludeSubs = ['s184', 's187', 's190', 's193', 's199', 's209', 's214', 's220', 's225', 's228', \
                       's234', 's240', 's213', 's235']

        subIDs = []

        for files in allDataFiles:
            if (files[16:] == 'expinfo.mat') and ((files[:4] not in excludeSubs) == True):
                subIDs.append(files[0:10])

    elif lvlAnalysis == 2:
        subIDs = []
        for files in allDataFiles:
            if files[16:] == 'expinfo.mat':
                subIDs.append(files[0:10])

    try:
        subIDs.remove('s193_ses2_')
        subIDs.remove('s193_ses1_')

    except:
        pass

    return subIDs


def compListsInd(list1, list2):
    final = [element for ind, element in enumerate(list1) if element in list2]
    ind = [ind for ind, element in enumerate(list1) if element in list2]
    return ind, final

### old scripts ###

# #for subID in subIDs2:
# #    ssvep_stimulus,ssvep_noise, ssvep_photocell = SSVEP_task3(subID)
# #    outname = path+subID+'_stimulus_SSVEP.mat'
# #    savemat(outname,)
#
# #%%
#
#
#
#
#
#     #    This is just trying to  fft each channel
#     #    trialestimate = np.zeros((4000,129,360))
#     #    for trial in finalgoodtrials:
#     #        trialdata = np.squeeze(data[:,:,trial])
#     #        trialdata = trialdata * np.transpose(weightsn)
#     #        trialestimate[:,:,trial] = trialdata[:,:]
#     #
#     #    # find the onset of noise for each trial
#     #    noise = pcdict['photonoise']
#     #    noisetrialestimate = np.zeros((1000,129,360))
#     #    for i in finalgoodtrials:
#     #        firstind = np.argmax(noise[:, i] > 0)
#     #        lastind = firstind + 1000
#     #        noisetrialestimate[:,:,i] = trialestimate[firstind:lastind,:,i]
#
#     #
#     #        # FFT the photocells
#     #    plt.subplot(322)
#     #    for x in range(0,numtrials):
#     #        y = p[1250:2249,x]
#     #        N = len(y)
#     #        yf = fft(y)
#     #        xf = np.linspace(0.0, sr/2, N//2)
#     #        plt.plot(xf[0:100], 2/N * np.abs(yf[0:100]))
#     #    plt.grid()
#     #    plt.title('Photocell Stimulus')
#     #
#     #        # FFT the noise
#     #    noisesignals = []
#     #    plt.subplot(324)
#     #    for x in range(0,numtrials):
#     #        firstind = np.argmax(n[:,x]>0)
#     #        lastind = firstind + 1500
#     #        noisesignal = n[firstind:lastind,x]
#     #        N = len(noisesignal)
#     #        yn = fft(noisesignal)
#     #        xn = np.linspace(0.0, sr/2, N//2)
#     #        plt.plot(xn[0:100], 2/N * np.abs(yn[0:100]))
#     #    plt.grid()
#     #    plt.title('Photocell Noise')
#     #
#     #        # FFT the for erp
#     #    plt.subplot(326)
#     #    for x in range(0,129):
#     #        y = erp[1250:2249,x]
#     #        N = len(y)
#     #        yf = fft(y)
#     #        xf = np.linspace(0.0, sr/2, N//2)
#     #        plt.plot(xf[0:60], 2/N * np.abs(yf[0:60]))
#     #        plt.title('ERP')
#     #    plt.grid()
#
#
# # %%
# # globals
# SSVEP = getSSVEP(subID)
#
# #%%
# ## %% Plots
# #
#  #photocell stimulus
#
#  plt.figure()
#  plt.subplot(411)
#  npc = len(yfp)
#  xfp = np.linspace(0.0, sr / 2, npc // 2)
#  plt.plot(xfp[0:100], 2 / npc * np.abs(yfp[0:100, :]))
#  plt.title('Photocell Stimulus using fft')
#  plt.xticks(np.arange(min(xfp[0:100]), max(xfp[0:100]) + 1, 10))
#
#  #photocell noise
#
#  plt.subplot(412)
#  nn = len(noisesignal)
#  plt.xticks(np.arange(min(xn[0:100]), max(xn[0:100]) + 1, 10))
#
#
#  #ERP after stim
#  plt.subplot(413)
#  nerp = len(yferp)
#  xf = np.linspace(0.0, sr / 2, (nerp // 2 + 1))
#  xfplot = xf[8*int(len(xf)/nyquist):51*int(len(xf)/nyquist)]
#  yfplot = yferp[8*int(len(xf)/nyquist):51*int(len(xf)/nyquist)]
#  plt.plot(xfplot, (2 * np.abs(yfplot)) ** 2)
#  plt.title('ERP (matched with onset of stimulus) for %i ms' % nerp)
#  plt.xticks(np.arange(min(xf[0:60]), max(xf[0:60]) + 1, 10))
#
#  #ERP after noise
# plt.subplot(414)
# nerpn = len(erpnoise)
# xferpn = np.linspace(0.0, sr / 2, nerpn // 2 +1)
# xferpnplot = xferpn[8*int(len(xferpn)/nyquist):51*int(len(xferpn)/nyquist)]
# yferpnplot = yferpn[8*int(len(xferpn)/nyquist):51*int(len(xferpn)/nyquist)]
# plt.plot(xferpnplot, (2 * np.abs(yferpnplot)) ** 2)
# plt.title('ERP fft(matched with onset of noise) for %i ms' % erpdur)
# plt.xticks(np.arange(min(xferpn[0:60]), max(xferpn[0:60]) + 1, 10))
#
# plt.tight_layout()
# plt.show
# #
# #    # optimal channel to detect noise (40Hz)
# #    # 35Hz-45Hz bin (1000ms)
# #    ferpn = fft(erpnoise, axis=0) # raw fft
# #    binwidth = int((len(erpnoise)/2 + 1)/nyquist)
# #    ferpn = ferpn[35*binwidth:45*binwidth +1]
# #    u,s,vh = linalg.svd(ferpn[:,goodchan])
# #
# #    weightsn = np.zeros((129,1),dtype=complex)
# #
# #	# This is an optimal set of weights to estimate 35-45hz signal.
# #    weightsn[goodchan,0] = np.matrix.transpose(vh[0,:])
# #
# #	# lets test it on the same interval using weighted electrode vs. original
# #    plt.figure()
# #    plt.subplot(311)
# #    yftest = fft(erpnoise, axis=0)
# #    erpnoisetest = np.matmul(yftest,weightsn)
# #    xferpn = np.linspace(0.0, nyquist, nerpn // 2 +1)
# #    plt.plot(xferpn[8:51], (2/len(erpnoise) * np.abs(erpnoisetest[8:51]) ** 2))
# #    plt.title('Testing the weighted channels on the ERP post-noise')
# #
# #
# #    plt.subplot(312)
# #    yerp = fft(erpnoise, axis=0)
# #    xferpn = np.linspace(0.0, nyquist, nerpn // 2 +1)
# #    plt.plot(xferpn[8:51], (2/len(erpnoise) * np.abs(yerp[8:51]) ** 2))
# #    plt.title('Original unweighted ERP')
# #
# #    Testing
# #   plt.figure()
# #    plt.subplot(311)
# #    xf = np.linspace(0.0, nyquist, nerp // 2 +1)
# #    plt.plot(xf[8:51], (2/len(erptest) * np.abs(erptest[8:51]) ** 2))
# #    plt.title('Testing the weighted channels on the ERP post-stimulus')
# #
# #
# #    plt.subplot(312)
# #    yferp = fft(y, axis=0)
# #    plt.plot(xf[8:51], (2/len(erptest) * np.abs(yferp[8:51]) ** 2))
# #    plt.title('Original Unweighted ERP')
# #
# #
# #    # fft the trial estimate 1000ms after noise (25-35hz bins used)
# #    plt.subplot(313)
# #    plt.plot(xf[25:51], (2/1000 * np.abs(trialestimatestim[25:51,0:10]) ** 2))
# #    plt.title('trial estimate for stim signals after fft using 25Hz-35Hz')
#
#
# #    # Now use the weights to loop through indivial trials
# #    noise = pcdict['photonoise']
# #    trialestimate = np.zeros((1000,360),dtype=complex)
# #    for trial in finalgoodtrials:
# #        trialdata = np.squeeze(data[:,:,trial])
# #        firstind = np.argmax(n[:, trial] > 0)
# #        lastind = firstind + 1000
# #        trialdata = trialdata[firstind:lastind,:]
# #        trialfft = fft(trialdata, axis = 0)
# #        trialweighted = np.matmul(trialfft,weightsn)
# #        trialestimate[:,trial] = trialweighted[:,0] #new time series
# #
# #    # fft the trial estimate 1000ms after noise (35-45hz bins)
# #    plt.subplot(313)
# #    xferpn = np.linspace(0.0, nyquist, nerp // 2 +1)
# #    plt.plot(xferpn[20:51], (2/1000 * np.abs(trialestimate[20:51,:]) ** 2))
# #    plt.title('trial estimate for noise signals after fft using 35Hz-45Hz')
# #
# #    plt.tight_layout()
# #    plt.show
#
#


