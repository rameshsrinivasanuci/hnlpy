#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 16:05:45 2020

@author: jenny
"""


'''this scripts is used to generate numpy arrays for '''

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

# picking the training set
subIDs = choose_subs(1, path)
subID =  's239_ses1_'    # use this subject for CORAL
subIDs.remove('s236_ses1_')
pca = read_mat('/home/jenny/pdmattention/task3/pcamodel_zscore/s181_ses1_pcamodel.mat')
rs_chans = pca['goodchannels']

subIDtrain = choose_subs(1, path)
subIDall = choose_subs(2, path)
subIDtest = [i for i in subIDall if i not in subIDtrain]


# some variables for adaptation


# get TransMatrx
def get_TransMatrix(freq):
    for index, sub in enumerate(subIDs):
        if sub !='s239_ses1_':
            print(index)
            Source, _, _, _, _, _ = trial_ssvep(sub, freq)
            transfor = CORAL()
            transfor.fit(Source, Target)
            Xs_trans = transfor.transfer(Source)  # adjusted source matrix
        else:
            Xs_trans = Target
            print(index)
    TransMatrix[:,:,index] = Xs_trans

# get goodtrial TransMatrix
def get_TransGoodTrials(freq):
    TargetMatrix, _,_,_,_ = trial_ssvep('s239_ses1_', freq)
    Target = []
    Data = np.empty((0,121))
    Target_rt = []
    Target_rawrt = []
    for index, sub in enumerate(subIDs):
        if sub !='s239_ses1_':
            print(index)
            Source, finalgoodtrials, acc, rt, rt_class = trial_ssvep(sub, freq)
            transfor = CORAL()
            transfor.fit(Source, TargetMatrix)
            Xs_trans = transfor.transfer(Source)  # adjusted source matrix
            Xs_trans = Xs_trans[:, finalgoodtrials].T
        else:
            print(index)
            Xs_trans = TargetMatrix
            _, finalgoodtrials, acc, rt, rt_class = trial_ssvep(sub, freq)
            Xs_trans = Xs_trans[:, finalgoodtrials].T
        Data = np.vstack((Data, Xs_trans))
        Target = np.append(Target, acc)
        Target_rt = np.append(Target_rt, rt_class)
        Target_rawrt = np.append(Target_rawrt, rt)
    return Data, Target, Target_rt, Target_rawrt

# get raw fft data coral corrected
Data30,Target,Target_rt, Target_rawrt =   (30)
Data40,_,_ = get_TransGoodTrials(40)

Dataraw = np.hstack((Data30,Data40))
np.save('/home/ramesh/pdmattention/ssvep/test/data_raw', Dataraw)



Data, Target, Target_rt = get_TransGoodTrials(40)

def get_singletrialN200():
    path = '/home/ramesh/pdmattention/task3/'
    Data = np.empty((0, 500))
    TargetMatrix = read_mat(path + 's239_ses1_' + 'N200.mat')['singletrial']
    for index, sub in enumerate(subIDs):
        if sub !='s239_ses1_':
            print(index)
            Source = read_mat(path + sub + 'N200.mat')['singletrial']
            transfor = CORAL()
            transfor.fit(Source, TargetMatrix)
            Xs_trans = transfor.transfer(Source)  # adjusted source matrix
            stimulus_ssvep, _, _, _, _, _, _ = SSVEP_task3(sub)
            finalgoodtrials = stimulus_ssvep['goodtrials']
            singletrial = Xs_trans[1250:1750, finalgoodtrials].T
        else:
            print(index)
            Xs_trans = TargetMatrix
            stimulus_ssvep, _, _, _, _, _, _ = SSVEP_task3(sub)
            finalgoodtrials = stimulus_ssvep['goodtrials']
            singletrial = Xs_trans[1250:1750, finalgoodtrials].T
        #
        # N200 = read_mat(path + sub + 'N200.mat')
        # singletrial = N200['singletrial']
        # stimulus_ssvep, _, _, _, _, _, _ = SSVEP_task3(sub)
        # finalgoodtrials = stimulus_ssvep['goodtrials']
        # singletrial = singletrial[1250:2000, finalgoodtrials].T
        Data = np.vstack((Data, singletrial))
np.save('/home/ramesh/pdmattention/ssvep/test/data_n200raw', Data)

def get_N200bychan():
    Data = np.empty((0, 500,119))
    path = '/home/ramesh/pdmattention/task3/'
    for index, sub in enumerate(subIDs):
        print(index, sub)
        stimulus_ssvep, _, _, _, _, _, behav = SSVEP_task3(sub)
        finalgoodtrials = stimulus_ssvep['goodtrials']
        data = read_mat(path + subID + 'task3_final.mat')['data'][1250:1750, rs_chans,:]
        data = data[:,:,finalgoodtrials]
        mean = np.tile(np.mean(data, axis=0), [500, 1, 1])
        datanew = data - mean
        correct = behav['acc']
        data = np.transpose(data, (2, 0, 1))
        Data = np.vstack((Data, data))
np.save('/home/ramesh/pdmattention/ssvep/test/data_n200raw', Data)

def get_N200bychanadapted():
    Data = np.empty((0, 500,119))
    path = '/home/ramesh/pdmattention/task3/'
    TargetMatrix = read_mat(path + subID + 'task3_final.mat')['data'][1250:1750, rs_chans,:]
    for index, sub in enumerate(subIDs):
        if sub !='s239_ses1_':
            print(index)
            stimulus_ssvep, _, _, _, _, _, behav = SSVEP_task3(sub)
            finalgoodtrials = stimulus_ssvep['goodtrials']
            Source = read_mat(path + subID + 'task3_final.mat')['data'][1250:1750, rs_chans,:]
            transfor = CORAL()
            transfor.fit(Source, TargetMatrix)
            Xs_trans = transfor.transfer(Source)  # adjusted source matrix
            singletrial = Xs_trans[1250:1750, finalgoodtrials].T

        data = data[:,:,finalgoodtrials]
        mean = np.tile(np.mean(data, axis=0), [500, 1, 1])
        datanew = data - mean
        correct = behav['acc']
        data = np.transpose(data, (2, 0, 1))
        Data = np.vstack((Data, data))


        if sub !='s239_ses1_':
            print(index)
            Source = read_mat(path + sub + 'N200.mat')['singletrial']
            transfor = CORAL()
            transfor.fit(Source, TargetMatrix)
            Xs_trans = transfor.transfer(Source)  # adjusted source matrix
            stimulus_ssvep, _, _, _, _, _, _ = SSVEP_task3(sub)
            finalgoodtrials = stimulus_ssvep['goodtrials']
            singletrial = Xs_trans[1250:1750, finalgoodtrials].T
        else:
            print(index)
            Xs_trans = TargetMatrix
            stimulus_ssvep, _, _, _, _, _, _ = SSVEP_task3(sub)
            finalgoodtrials = stimulus_ssvep['goodtrials']
            singletrial = Xs_trans[1250:1750, finalgoodtrials].T


np.save('/home/ramesh/pdmattention/ssvep/test/data_n200raw', Data)




def get_N200param():
    path = '/home/ramesh/pdmattention/task3/'
    Data = np.empty((0, 2))
    for index, sub in enumerate(subIDs):
        print(index, sub)
        N200 = read_mat(path + sub + 'N200.mat')
        peakvalue = N200['peakvalue']
        peaktime = N200['peaktiming']
        erp_peakvalue = N200['erp_peakvalue']
        erp_peaktime = N200['erp_peaktiming']
        relative_peakvalue = peakvalue - erp_peakvalue
        relative_peaktime = peaktime - erp_peaktime
        stimulus_ssvep, _, _, _, _, _, _ = SSVEP_task3(sub)
        finalgoodtrials = stimulus_ssvep['goodtrials']
        relative_peakvalue = relative_peakvalue[finalgoodtrials]
        relative_peaktime = relative_peaktime[finalgoodtrials]
        n200_param = np.vstack((relative_peakvalue, relative_peaktime)).T
        Data = np.vstack((Data, n200_param))
np.save('/home/ramesh/pdmattention/ssvep/test/data_n200raw', Data)



def get_unTransMatrix():
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

np.save('/home/ramesh/pdmattention/ssvep/test/data_real', Data)

# make a lowpass filter
sr = 1000
sos, w, h = timeop.makefiltersos(sr, 8, 10)
erpfilt = signal.sosfiltfilt(sos, n200, axis=1, padtype='odd')
