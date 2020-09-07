# this scripts preprocess the Ostwald data
# this function imports the data, epoch all trials, baseline correction
# trial and channel rejection, ICA, re-reference data using average and baseline correction

# example of input
# subID = 'sub-003'
# run = '02'

from scipy.io import savemat
from pymatreader import read_mat
import numpy as np
import timeop
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy import linalg
import timeop
import diffusion
import os
import numpy.matlib
from collections import Counter
import get_erp_ostwald
import imagesc
import get_erp_ostwald as geo

path = '/home/jenny/ostwald-data/clean-eeg-converted/'


def get_allerp():
    subIDs, subIDs_run = geo.get_sub(path)
    for sub in subIDs:
        granddict = grandmean_ERP(sub)
        artifact = granddict['artifact']
        granderp = granddict['granderp']
        goodchan = granddict['goodchan']
        goodtrials = granddict['goodtrials']
        badchan = granddict['badchan']
        badtrials = granddict['badtrials']

        imagesc.plot(artifact, cbar = False)
        plt.title('%s' % sub)
        plt.text(0.5, -10, 'excluded_channels for grand: %s' % len(badchan))
        plt.text(0.5, -5, 'excluded_trials for grand: %s' % len(badtrials))

        plt.figure()
        plt.plot(np.arange(-100, 700, 2), granderp[950:1350, goodchan])
        plt.title('%s' % sub)

        plt.savefig('/home/jenny/ostwald-data/clean-eeg-converted/' + '/Figures/ERPs/cleanERP-sub%s.png' % sub[4:])


def grandmean_ERP(subID):
    run1 = get_epoch(subID,'01')
    run2 = get_epoch(subID,'02')
    run3 = get_epoch(subID,'03')
    run4 = get_epoch(subID,'04')
    if subID != 'sub-001':
        run5 = get_epoch(subID,'05')

    grand = []

    grand = np.append(run1['trialeeg'],run2['trialeeg'],axis = 2)
    grand = np.append(grand, run3['trialeeg'],axis = 2)
    grand = np.append(grand, run4['trialeeg'],axis = 2)
    if subID !='sub-001':
        grand = np.append(grand, run5['trialeeg'],axis = 2)

    condition = []

    if subID != 'sub-001':
        condition = list(np.array(run1['condition'])[:,1]) + list(np.array(run2['condition'])[:,1]) + \
        list(np.array(run3['condition'])[:,1]) + list(np.array(run4['condition'])[:,1]) + \
        list(np.array(run5['condition'])[:,1])

    # construct a channel x trial matrix
    channelnum = grand.shape[1]
    trialnum = grand.shape[2]
    artifact = np.zeros((channelnum, trialnum))

    # for each channel in each trial, mark 1 if the abs(mV) is more than 100
    for trial in range(0, trialnum):
        for chan in range(0, channelnum):
            waveform = grand[:, chan, trial]
            if all(abs(i) <= 100 for i in waveform) is True:
                artifact[chan, trial] = 0
            else:
                artifact[chan, trial] = 1

    goodtrialsgrand, goodchangrand = get_indices(artifact)
    excluded_trialsgrand = [i for i in range(0,trialnum) if i not in goodtrialsgrand]
    excluded_channelsgrand = [i for i in range(0,channelnum) if i not in goodchangrand]
    # imagesc.plot(artifact, cbar = False)
    # plt.title('%s' % subID)
    print('excluded_trials for grand: %s' % len(excluded_trialsgrand))
    print('excluded_channels for grand: %s' % len(excluded_channelsgrand))


    erpall = np.mean(grand[:, :, goodtrialsgrand], axis=2)

    # make a lowpass filter
    # sr = 500
    # sos, w, h = timeop.makefiltersos(sr, 10, 20)
    # erpfilt = signal.sosfiltfilt(sos, erpall, axis=0, padtype='odd')

    # make a highpass filter
    # sos, w, h = timeop.makefiltersos(sr, 1, 0.5)
    # sos, w, h = timeop.makefiltersos(sr, 0.1, 0.05)
    # sos,w,h = timeop.makefiltersos(sr,0.5,0.25)
    # newfilt = signal.sosfiltfilt(sos, erpfilt, axis=0, padtype='odd')

    # erpfiltbaseall = timeop.baselinecorrect(erpfilt, np.arange(948, 998, 1))


    grandmean = dict()
    grandmean = {'granderp': erpfiltbaseall, 'goodtrials': goodtrialsgrand, \
              'goodchan': goodchangrand, 'badtrials': excluded_trialsgrand, \
              'badchan':excluded_channelsgrand, 'condition': condition, 'artifact': artifact}
    return grandmean


def get_epoch(subID, run):
# this function gets epochs based on stimulus
# returns eegdata, eeg plus ecg+eog, artifact matrix
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

    all_stim = left_hcp + left_hcnp + left_lcp + right_lcnp\
               + right_hcp + right_hcnp + right_lcp + right_lcnp
    all_stim.sort()

    # index the latency, minus one because times series starts at 0 index
    tstim = latency[all_stim]
    tstim = tstim -1

    # transpose the data
    data = np.transpose(data)

    # eeg = np.delete(data, slice(30, 32), axis=1)

    # construct a time x channel x trial matrix for each run for 6s
    samples = int(5*sr)
    channelnum = data.shape[1]
    trialnum = tstim.shape[0]
    trialdata = np.zeros((samples,channelnum, trialnum))

    # epoch the data to create single-trial segments for 5s
    for i in np.arange(trialnum):
        time = tstim[i]
        trialdata[:,:, i] = data[time-1000: time+1500,:]

    # remove EOG and ECG and get trialeeg
    trialeeg = np.delete(trialdata, slice(30, 32), axis=1)

    # baseline correction
    # get the mean of 100ms pre-stim and remove it from the whole window
    for i in range(0, trialnum):
        baseline_mean = np.tile(np.mean(trialeeg[(1000 - 50):1000, :, i], axis=0), [trialeeg.shape[0], 1])
        trialeeg[:, :, i] = trialeeg[:, :, i] - baseline_mean

    # construct a channel x trial matrix
    channelnum = trialeeg.shape[1]
    artifact = np.zeros((channelnum, trialnum))

    # for each channel in each trial, mark 1 if the abs(mV) is more than 100
    for trial in range(0, trialnum):
        for chan in range(0, channelnum):
            waveform = trialeeg[:,chan,trial]
            if all(abs(i)<=100 for i in waveform) is True:
                artifact[chan, trial] = 0
            else:
                artifact[chan, trial] = 1

    # get the indices for the good channels and trials
    goodtrials, goodchan = get_indices(artifact)

    excluded_trials = [i for i in range(0,trialnum) if i not in goodtrials]
    excluded_channels = [i for i in range(0,channelnum) if i not in goodchan]
    print('excluded_trials: %s' % len(excluded_trials))
    print('excluded_channels: %s' % len(excluded_channels))

    # save to dataDict
    dataDict = {
        'trialeeg': trialeeg,
        'trialdata': trialdata,
        'data': data,
        'artifact': artifact,
        'goodtrials': goodtrials,
        'goodchan': goodchan,
        'bad_trials': excluded_trials,
        'bad_chan': excluded_channels,
        'cleandata': trialeeg,
        'condition': cond_ind,
        'tstim': tstim
        }
    filename = '/home/jenny/ostwald-data/clean-eeg-converted/' + subID + \
               '_run-' + run + '_info' + '.mat'
    savemat(filename, dataDict)
    return dataDict

def get_indices(artifact):
    artifact0 = artifact.sum(axis = 0)
    artifact1 = artifact.sum(axis = 1)

    epoch_threshold = int(artifact.shape[1] * 0.2)
    channel_threshold = int(artifact.shape[0] * 0.2)

    goodtrials = np.squeeze(np.array(np.where(artifact0 < channel_threshold)))
    goodchan = np.squeeze(np.array(np.where(artifact1 < epoch_threshold)))

    return goodtrials, goodchan


