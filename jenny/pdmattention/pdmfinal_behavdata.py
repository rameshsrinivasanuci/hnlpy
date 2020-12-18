"""
# Created on 12/16/20 5:48 PM 2020

# author: Jenny Sun
"""

"""this script is used for generating ssvep and spectragram data
for linking alpha desynchronization and HDDM by Micheal using block of 60 trials
the general procedure follows the .m script under ./pdmattention/DecisionChronometrics/Analysiis/...
...dataprep/pdmfinal_behavdata.m"""
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
# import lab modules
import timeop
from ssvep_subject import *

path = ('/home/jenny/pdmattention/')

# getting the behav subject info and parameter estimates
alphadc_info = read_mat(path +'DecisionChronometrics/Data/modelfits/behavior_task3_Dec_15_20_13_21.mat' )
alphadc_est = read_mat(path + 'DecisionChronometrics/Data/modelfits/behavior_task3_HDDM_AlphaDCDec_10_20_10_11_estimates.mat')

subIDs = alphadc_info['uniquepart']
subIDs = ['s' + str(i) for i in subIDs]
fullIDs = choose_subs(1, path + 'task3')
fullIDs.append('s185_ses2_')
fullIDs.remove('s236_ses1_')
fullIDs.sort()
check = all([True for i, j in enumerate(fullIDs) if str(j)[0:4] == subIDs[i]])
while check:
    print('subIDs matched')
    check = False


# subID =  's239_ses1_'    # use this subject for CORAL

pca = read_mat('/home/jenny/pdmattention/task3/pcamodel_zscore/s181_ses1_pcamodel.mat')
rs_chans = pca['goodchannels']


###

def get_ssvepAll(freq1,freq2):
    """this function get SSVEPs by 30 and 40 Hz for all
    freq1 is the power we want to extract from the ssvep maximizing 30Hz
    freq2 is the power we want to extract from the ssvep maximizing 40Hz"""
    path = '/home/ramesh/pdmattention/task3/'
    Signal30 = np.empty((0,4))
    Signal30 = np.empty((0,4))
    for index, sub in enumerate(subIDs):
        print(index)
        if sub == 's185_ses2_':
            stimulus_ssvep, noise_ssvep, behavdict = SSVEP_185(subID)
            datadict = read_mat(path + subID + 'task3_cleaned.mat')
        else:
            stimulus_ssvep, noise_ssvep, _, behavdict, behavfinal = SSVEP_task3All(subID)
            condition = read_mat(path + subID + 'task3_expinfo.mat')['condition']
            datadict  = read_mat(path  + subID + 'task3_final.mat')
        artifact = datadict['artifact']
        nchans = datadict['data'].shape[1]
        ntrials = datadict['data'].shape[2]
        artifacttrials = sum(artifact) == nchans
        goodchannels = stimulus_ssvep['goodchannels']
        # this is to match michael's code
        goodtrials = np.where(artifacttrials == False)

        # signal_power30 is the maximizing 30Hz ssvep results
        signal_30 = stimulus_ssvep['trial_allchan'][:,goodtrials[0]]
        signal_40 = noise_ssvep['trial_allchan'][:,goodtrials[0]]

        # get the power at 30Hz and 40Hz (signal) and 29Hz + 31Hz (noise) from all channels
        signal_power30 = 2 * np.abs(signal_30[freq1, :]) ** 2
        noise_power30 = 2 * (1 / 2 * (np.abs(signal_30[freq1 - 1, :]) ** 2 + np.abs(signal_30[freq1 + 1, :]) ** 2))

        signal_power40 = 2 * np.abs(signal_40[freq2, :]) ** 2
        noise_power40 = 2 * (1 / 2 * (np.abs(signal_40[freq2 - 1, :]) ** 2 + np.abs(signal_40[freq2 + 1, :]) ** 2))

        # get the snr, the division ==0 for all photocell channels
        snr30 = np.divide(signal_power30, noise_power30, out=np.zeros_like(signal_power30), where=noise_power30 != 0)
        snr40 = np.divide(signal_power40, noise_power40, out=np.zeros_like(signal_power40), where=noise_power40 != 0)

        condition = condition[goodtrials[0]]
        nt = len(goodtrials[0])
        true_participant =np.tile(int(subID[1:4]), (1, nt))
        trialdata30 = np.zeros((nt, 4),  dtype=float)
        trialdata40 = np.zeros((nt, 4),  dtype=float)
        trialdata30 = np.vstack((snr30,signal_power30, condition, np.squeeze(true_participant))).T
        trialdata40 = np.vstack((snr40, signal_power40, condition, np.squeeze(true_participant))).T

        Signal30 = np.vstack((Signal30, trialdata30))
        Signal40 = np.vstack((Signal40, trialdata40))




    Target = []
    Data = np.empty((0,242))
    Target_rt = []
    Target_rawrt =[]
    for index, sub in enumerate(subIDs):
        print(index)
        StimSnr30, finalgoodtrials, acc, rt, rt_class = trial_ssvep(sub, 30)
        StimSnr40, _, _, _, _ = trial_ssvep(sub, 40)
        StimSnrall = np.hstack((StimSnr30.T, StimSnr40.T))
        StimSnrall = StimSnrall[finalgoodtrials,:]
        Data = np.vstack((Data, StimSnrall))
        Target = np.append(Target, acc)
        Target_rt = np.append(Target_rt, rt_class)
        Target_rawrt = np.append(Target_rt, rt)


def SSVEP_task3All(subID):
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

    # time window of interest
    window = [1250, 2250]

    # FFT the eeg data
    stimulus_ssvep = getSSVEP(data,sr,window,30,np.arange(0,360,1),rs_chans)
    noise_ssvep = getSSVEP(data,sr,window,40,np.arange(0,360,1),rs_chans)
    #
    # # same procedure analyzed by condition
    # ind_ez = np.where(condition_final == 1)
    # ind_md = np.where(condition_final == 2)
    # ind_hr = np.where(condition_final == 3 )
    #
    # stim_ez = getSSVEP(data,sr,window,30, finalgoodtrials[ind_ez], goodchans)
    # stim_md = getSSVEP(data, sr, window, 30, finalgoodtrials[ind_md], goodchans)
    # stim_hr = getSSVEP(data, sr, window, 30, finalgoodtrials[ind_hr], goodchans)
    # noise_ez = getSSVEP(data,sr,window, 40, finalgoodtrials[ind_ez], goodchans)
    # noise_md = getSSVEP(data, sr, window, 40, finalgoodtrials[ind_md], goodchans)
    # noise_hr = getSSVEP(data, sr, window, 40, finalgoodtrials[ind_hr], goodchans)

    # stim_erpf = np.dstack((stim_ez['erp_fft'], stim_md['erp_fft'], stim_hr['erp_fft']))
    # noise_erpf = np.dstack((noise_ez['erp_fft'], noise_md['erp_fft'], noise_hr['erp_fft']))

    return stimulus_ssvep, noise_ssvep, photocell, behavdict, behav_final





def SSVEP_185(subID):
    currentSub = subID[0:4]
    print('Current Subject: ', currentSub)
    datadict = read_mat(path  + subID + 'task3_cleaned.mat')
    behavdict = datadict['expinfo']

    data = np.array(datadict['data'])
    artifact = np.array(datadict['artifact'])
    sr = np.array(datadict['sr'])
    rt = behavdict['rt']
    correct = behavdict['correct']

    # open up indices
    artifact0 = artifact.sum(axis=0)
    artifact1 = artifact.sum(axis=1)

    # identify goodtrials and good channels.
    goodtrials = np.squeeze(np.array(np.where(artifact0 < 20)))
    goodchans = np.squeeze(np.array(np.where(artifact1 < 40)))
    #
    # # choosing the trials that have RT over 300ms and check if they are all goodtrials
    # # get the index and apply to rt, condition and accuracy
    # xy, x_ind, y_ind = np.intersect1d(beh_ind, goodtrials, return_indices=True)
    # ind, finalgoodtrials = np.array(compListsInd(beh_ind, goodtrials))
    # # here the ind stands for ind in beh_ind that are qualified as goodtrials
    # # finalgoodtrials stand for the actual trial index from the original unsorted dataset
    # rt_final = rt[ind]
    # acc_final = correct[ind]
    # behav_final = {'rt': rt_final, 'acc': acc_final}
    # condition_final = condition[ind]
    # correct_final = correct[ind]

    # time window of interest
    window = [1250, 2250]

    # FFT the eeg data
    stimulus_ssvep = getSSVEP(data,sr,window,30,np.arange(0,360,1),goodchans)
    noise_ssvep = getSSVEP(data,sr,window,40,np.arange(0,360,1),goodchans)
    #
    # # same procedure analyzed by condition
    # ind_ez = np.where(condition_final == 1)
    # ind_md = np.where(condition_final == 2)
    # ind_hr = np.where(condition_final == 3 )
    #
    # stim_ez = getSSVEP(data,sr,window,30, finalgoodtrials[ind_ez], goodchans)
    # stim_md = getSSVEP(data, sr, window, 30, finalgoodtrials[ind_md], goodchans)
    # stim_hr = getSSVEP(data, sr, window, 30, finalgoodtrials[ind_hr], goodchans)
    # noise_ez = getSSVEP(data,sr,window, 40, finalgoodtrials[ind_ez], goodchans)
    # noise_md = getSSVEP(data, sr, window, 40, finalgoodtrials[ind_md], goodchans)
    # noise_hr = getSSVEP(data, sr, window, 40, finalgoodtrials[ind_hr], goodchans)
    #
    # stim_erpf = np.dstack((stim_ez['erp_fft'], stim_md['erp_fft'], stim_hr['erp_fft']))
    # noise_erpf = np.dstack((noise_ez['erp_fft'], noise_md['erp_fft'], noise_hr['erp_fft']))

    return stimulus_ssvep, noise_ssvep, behavdict
