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
import matplotlib.patches as mpatches
# import lab modules
import timeop
from ssvep_subject import *

path = ('/home/jenny/pdmattention/')

# getting the behav subject info and parameter estimates
alphadc_info = read_mat(path +'DecisionChronometrics/Data/modelfits/behavior_task3_Dec_15_20_13_21.mat' )
alphadc_est = read_mat(path + 'ssvep/behavior2_task3_HDDM_DCDec_18_20_18_19_estimates.mat')


subIDs = alphadc_info['uniquepart']
subIDs = np.delete(subIDs, 3)
subIDs = ['s' + str(i) for i in subIDs]
fullIDs = choose_subs(1, path + 'task3')
# fullIDs.append('s185_ses2_')
fullIDs.remove('s236_ses1_')
fullIDs.sort()
check = all([True if str(j)[0:4] == subIDs[i] else False for i, j in enumerate(fullIDs)])
if check:
    print('subIDs matched')
else:
    print('Warning: subID not matched!')

# run original model where delta is fixed
model1delta = np.zeros((len(subIDs),3), dtype=float)
for i in range(0,33):
    sub = subIDs[i]
    delta = read_mat('/home/mariel/Documents/Projects2/subs_genparam/' + sub + 'diags.mat' )['delta']['mean']
    model1delta[i,:] = delta



# plots

Signal30 = np.load('/home/jenny/pdmattention/ssvep/block_hddm/Signal30.npy')
Signal40 = np.load('/home/jenny/pdmattention/ssvep/block_hddm/Signal40.npy')
Signal30[Signal30[:,2] ==4, 2] = np.ones(1961);
Signal30[Signal30[:,2] ==5, 2] = 2 * np.ones(1961);
Signal30[Signal30[:,2] ==6, 2] = 3 * np.ones(1968);

cond = ['cond1','cond2','cond3']
color = ['b','g','r']
marker = ['^','*','o']

power30_sub = np.zeros((npart,3), dtype=float)
for i in range(0,npart):
    sub = uniquepart[i]
    subdata = Signal30[list(np.where(Signal30[:, 3] == sub)[0]), :]
    for j in range(0,3):
        power30_sub [i,j] = np.mean(subdata[list(np.where(subdata[:,2] == j+1)[0]),1])

for i in range(0,npart):
    for j in range(0,3):
        plt.scatter(deltanew[i,j],np.log(power30_sub[i,j]), color = color[j],\
        marker = marker[j], s = 50)
plt.title('Power at 30Hz (fixed alpha)')
plt.xlabel('Delta')
plt.ylabel('Log of Power')

easy = mlines.Line2D([], [],color = 'b',marker='^', label = 'easy')
medium = mlines.Line2D([], [],color = 'g', marker='*',label = 'medium')
hard = mlines.Line2D([], [],color = 'r', marker='o',label = 'hard')
plt.legend(handles=[easy, medium, hard])



cond = ['cond1','cond2','cond3']
color = ['b','g','r']
marker = ['^','*','o']
for i in range(0,3):
    plt.scatter(model1delta[i], np.squeeze(power40)[i], color = color[i], label = cond[i],\
    marker = marker[i], s = 50)

plt.title('Power at 40Hz')
plt.xlabel('SNR')
plt.ylabel('Varsig')
plt.legend(loc='best')

plt.scatter(varsigmahier['mean'],np.squeeze(power30), color='r', marker = 'o', lable='block1')

# compared with fixed alpha data
deltanew = np.zeros((33,3), dtype=float)
delta = alphadc_est['delta']
delta = delta['mean']
deltanew[:,0] = 0.5*(delta[:,0] + delta[:,3])
deltanew[:,1] = 0.5*(delta[:,1] + delta[:,4])
deltanew[:,2] = 0.5*(delta[:,2] + delta[:,5])



# subID =  's239_ses1_'    # use this subject for CORAL

pca = read_mat('/home/jenny/pdmattention/task3/pcamodel_zscore/s181_ses1_pcamodel.mat')
rs_chans = pca['goodchannels']


Signal30 = np.load('/home/jenny/pdmattention/ssvep/block_hddm/Signal30.npy')
Signal40 = np.load('/home/jenny/pdmattention/ssvep/block_hddm/Signal40.npy')



#






###
# get rid of sub185

fullIDs.remove('s185_ses2_')

Signal30, Signal40 = get_ssvepAll(30,40, fullIDs)
np.save('/home/ramesh/pdmattention/ssvep/block_hddm/Signal30', Signal30)
np.save('/home/ramesh/pdmattention/ssvep/block_hddm/Signal40', Signal40)


# get the data



def get_ssvepAll(freq1,freq2, subIDs):
    """this function get SSVEPs by 30 and 40 Hz for all
    freq1 is the power we want to extract from the ssvep maximizing 30Hz
    freq2 is the power we want to extract from the ssvep maximizing 40Hz"""
    path = '/home/ramesh/pdmattention/task3/'
    Signal30 = np.empty((0,4))
    Signal40 = np.empty((0,4))
    Signal40 = np.empty((0,4))
    for index, sub in enumerate(subIDs):
        print(index)
        print(sub)
        if sub == 's185_ses2_':
            stimulus_ssvep, noise_ssvep, behavdict = SSVEP_185(sub)
            datadict = read_mat(path + sub + 'task3_cleaned.mat')
            condition = datadict['expinfo']['condition']
        else:
            stimulus_ssvep, noise_ssvep,behavdict, behavfinal = SSVEP_task3All(sub)
            condition = read_mat(path + sub + 'task3_expinfo.mat')['condition']
            datadict  = read_mat(path  + sub + 'task3_final.mat')
        thevars = np.var(datadict['data'], axis=0)
        artifact = thevars == 0 | np.isnan(thevars)
        if 'artifact' in datadict:
            artifact = datadict['artifact'] | artifact
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
        true_participant =np.tile(int(sub[1:4]), (1, nt))
        trialdata30 = np.zeros((nt, 4),  dtype=float)
        trialdata40 = np.zeros((nt, 4),  dtype=float)
        trialdata30 = np.vstack((snr30,signal_power30, condition, np.squeeze(true_participant))).T
        trialdata40 = np.vstack((snr40, signal_power40, condition, np.squeeze(true_participant))).T

        Signal30 = np.vstack((Signal30, trialdata30))
        Signal40 = np.vstack((Signal40, trialdata40))
    return Signal30, Signal40


def SSVEP_task3All(subID):
    path = '/home/ramesh/pdmattention/task3/'
    currentSub = subID[0:4]
    print('Current Subject: ', currentSub)
    # pcdict = read_mat(path + subID + 'task3_photocells.mat')
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

    return stimulus_ssvep, noise_ssvep, behavdict, behav_final





def SSVEP_185(subID):
    path = '/home/ramesh/pdmattention/task3/'
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





# plots

Signal30 = np.load('/home/jenny/pdmattention/ssvep/block_hddm/Signal30.npy')
Signal40 = np.load('/home/jenny/pdmattention/ssvep/block_hddm/Signal40.npy')



cond = ['cond1','cond2','cond3','cond4','cond5','cond6']
color = ['b','g','r','c','m','y']
marker = ['^','*','o', '^','*','o']
for i in range(0,6):
    plt.scatter(varsigmahier['mean'][i], np.squeeze(power40)[i], color = color[i], label = cond[i],\
    marker = marker[i], s = 50)

plt.title('Power at 40Hz')
plt.xlabel('SNR')
plt.ylabel('Varsig')
plt.legend(loc='best')

plt.scatter(varsigmahier['mean'],np.squeeze(power30), color='r', marker = 'o', lable='block1')


############################ by subject ######################################
varsigma = alphadc_est['varsigma']
ndt = alphadc_est['ndt']
true_participant = alphadc_info['true_participant']
uniquepart = alphadc_info['uniquepart']
uniquepart = np.delete(uniquepart, 3, 0)
delta = alphadc_est['delta']
npart =len(uniquepart)


cond = ['cond1','cond2','cond3','cond4','cond5','cond6']
color = ['b','g','r','c','m','y']
marker = ['^','*','o', '^','*','o']
power40_sub = np.zeros((npart,6), dtype=float)
for i in range(0,npart):
    sub = uniquepart[i]
    subdata = Signal30[list(np.where(Signal30[:, 3] == sub)[0]), :]
    for j in range(0,6):
        power40_sub [i,j] = np.mean(subdata[list(np.where(subdata[:,2] == j+1)[0]),0])

for i in range(0,npart):
    for j in range(0,6):
        plt.scatter(ndt['mean'][i,j],np.log(power40_sub[i,j]), color = color[j],\
        marker = marker[j], s = 50)
plt.title('SNR at 30Hz (fixed Alpha)')
plt.xlabel('NDT')
plt.ylabel('Log of snr')
plt.legend(loc='best')



for i in range(0,6):
    plt.scatter(varsigmahier['mean'][i], np.squeeze(power40)[i], color = color[i], label = cond[i],\
    marker = marker[i], s = 50)

plt.title('Power at 40Hz')
plt.xlabel('SNR')
plt.ylabel('Varsig')
plt.legend(loc='best')

plt.scatter(varsigmahier['mean'],np.squeeze(power30), color='r', marker = 'o', lable='block1')


snr30 = np.zeros((1,6), dtype=float)
snr30[0,0] = np.mean(Signal30[list(np.where(Signal30[:,2] == 1)[0]),0])
snr30[0,1] = np.mean(Signal30[list(np.where(Signal30[:,2] == 2)[0]),0])
snr30[0,2] = np.mean(Signal30[list(np.where(Signal30[:,2] == 3)[0]),0])
snr30[0,3] =np.mean(Signal30[list(np.where(Signal30[:,2] == 4)[0]),0])
snr30[0,4] =np.mean(Signal30[list(np.where(Signal30[:,2] == 5)[0]),0])
snr30[0,5] =np.mean(Signal30[list(np.where(Signal30[:,2] == 6)[0]),0])


snr40 = np.zeros((1,6), dtype=float)
snr40[0,0] = np.mean(Signal40[list(np.where(Signal40[:,2] == 1)[0]),0])
snr40[0,1] = np.mean(Signal40[list(np.where(Signal40[:,2] == 2)[0]),0])
snr40[0,2] = np.mean(Signal40[list(np.where(Signal40[:,2] == 3)[0]),0])
snr40[0,3] =np.mean(Signal40[list(np.where(Signal40[:,2] == 4)[0]),0])
snr40[0,4] =np.mean(Signal40[list(np.where(Signal40[:,2] == 5)[0]),0])
snr40[0,5] =np.mean(Signal40[list(np.where(Signal40[:,2] == 6)[0]),0])

power40 = np.zeros((1,6), dtype=float)
power40[0,0] = np.mean(Signal40[list(np.where(Signal40[:,2] == 1)[0]),1])
power40[0,1] = np.mean(Signal40[list(np.where(Signal40[:,2] == 2)[0]),1])
power40[0,2] = np.mean(Signal40[list(np.where(Signal40[:,2] == 3)[0]),1])
power40[0,3] =np.mean(Signal40[list(np.where(Signal40[:,2] == 4)[0]),1])
power40[0,4] =np.mean(Signal40[list(np.where(Signal40[:,2] == 5)[0]),1])
power40[0,5] =np.mean(Signal40[list(np.where(Signal40[:,2] == 6)[0]),1])
