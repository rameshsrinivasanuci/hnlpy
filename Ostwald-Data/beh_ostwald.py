# module for loading and preprocessing the Ostwald behavioral data
import pandas as pd
from get_erp_ostwald import get_sub
from scipy.io import savemat
import os
from pymatreader import read_mat
from collections import defaultdict
import numpy as np



base_dir = '/home/jenny/ostwald-data/'
targetpath = '/home/jenny/ostwald-data/inside-beh-converted/'


# this function saves all the subject behaviral data to the target path
# it also returns a dictionary where it contains the file name for each run
def save_behavlist():
    subIDs, _ = get_sub(base_dir)
    subIDs = list(subIDs)
    subIDs.remove('sub-001')
    subdict = dict()
    for sub in subIDs:
        subfile = behav_list(base_dir, sub)
        subdict[sub] = subfile
    return subdict

def combine_runs():
    subdict = save_behavlist()
    count = 0
    for sub in subdict:
        allrunsdict = subdict[sub]
        newRT = []
        newCond = []
        newACC = []
        count = count + 1
        newY = []
        for run in allrunsdict:
            filename = str(run)
            datadict = read_mat(filename)
            rts = datadict['rt']
            condition = datadict['condition']
            correct = datadict['correct']
            y = datadict['rt']  * np.sign(datadict['correct'] - 1 / 2)

            newRT = newRT + rts.tolist()
            newCond = newCond + condition.tolist()
            newACC = newACC + correct.tolist()
            newY = newY + y.tolist()

        sub_ind = [count] * len(newRT)
        dataDict = {'condition': newCond, 'rt': newRT, 'correct': newACC, 'y': newY, \
            'participant': sub_ind}
        filename = '/home/jenny/ostwald-data/inside-beh-converted/combined/' + sub + '-beh-concat' \
                     + '.mat'
        savemat(filename, dataDict)


def behav_list(base_dir, sub):
    subpath = base_dir + sub + '/sourcedata-eeg_inside-MRT/beh/'
    dataList = os.listdir(subpath)
    datafiles = []
    datafiles = [i for i in dataList if '.tsv' in i]
    datafiles.sort()
    subfiles = []
    for file in datafiles:
        dataname = subpath + file
        currentRun = dataname[-14:-8]
        tsv_read = pd.read_csv(dataname, sep='\t')

        # Identify the invalid trials
        tsv_read = tsv_read.drop([tsv_read.index[0], tsv_read.index[-1]])
        tsv_read = tsv_read.reset_index(drop=True)
        tsv_read = tsv_read.reset_index(level=0)
        tsv_read = tsv_read.rename(columns={'index': 'trial'})
        excluded = np.where(np.isnan(tsv_read['response_time']))[0]
        slow_trials =  [i for i, c in enumerate(tsv_read['response_time']) if c <= 0.3]
        excluded = np.append(excluded,slow_trials)
        tsv_read = tsv_read.drop(tsv_read.index[excluded.astype(int)], axis=0)

        # rename some of the keys
        tsv_read = tsv_read.to_dict("list")
        tsv_read['rt'] = tsv_read.pop('response_time')
        tsv_read['correct'] = tsv_read.pop('response_corr')

        # save the files
        filename = targetpath + sub + '-inside-' + dataname[-14:-4] + '.mat'
        savemat(filename, tsv_read)
        print('trials excluded: ' + str(len(excluded)))
        print(sub + ' ' + currentRun + ' saved')
        subfiles.append(filename)
    return subfiles

def AllBehavFile():
    subIDs, _ = get_sub(base_dir)
    subIDs = list(subIDs)
    subIDs.remove('sub-001')

    newRT = []
    newCond = []
    newACC = []
    newY = []
    newSub = []

    for subject in subIDs:
        datadict = read_mat(targetpath + '/combined/' + subject + '-beh-concat.mat')
        rts = datadict['rt']
        condition = datadict['condition']
        correct = datadict['correct']
        y = datadict['y']
        sub = datadict['participant']

        newRT = newRT + rts.tolist()
        newCond = newCond + condition.tolist()
        newACC = newACC + correct.tolist()
        newY = newY + y.tolist()
        newSub = newSub + sub.tolist()
    ncond = 4
    nparts = len(subIDs)
    N = len(newRT)
    dataDict = {'condition': newCond, 'rt': newRT, 'acc': newACC, 'y': newY, \
                'participant': newSub, 'nconds': ncond, 'nparts': nparts}
    filename = targetpath + 'allsubtest1.mat'
    savemat(filename, dataDict)
    return filename