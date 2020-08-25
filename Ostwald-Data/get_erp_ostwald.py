from pymatreader import read_mat
import numpy as np
import timeop
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.io import savemat
import timeop
import diffusion
import os
import numpy.matlib
from collections import Counter

subID = 'sub-003'
run = '02'
path = '/home/jenny/ostwald-data/clean-eeg-converted/'

def main():
    get_allerp()

def get_allerp():
    subIDs, subIDs_run = get_sub(path)

    for sub in subIDs:
        numofrun = subIDs_run[sub]
        run_index = np.arange(1,numofrun+1)
        run_index = [str('0' + str(i)) for i in run_index]
        # plt.figure(figsize=(20, 8))
        plt.figure()
        plt.title('ERP for subject %s (all runs)' % sub[4:])

        erp_all = np.zeros((600, 62, numofrun))
        for run in run_index:
            erp_all[:, :, int(run)-1] = get_erp(sub, run)
            mean_erp =np.mean(erp_all, axis=2)

        if sub == 'sub-012':
             mean_erp = data = np.delete(mean_erp, 23, axis = 1)

    # #  remove the mean when plotting
    #      epochlength = mean_erp.shape[0]
    #      erpmean = np.tile(np.mean(mean_erp, axis=0), [epochlength, 1])
    #      mean_erp = mean_erp - erpmean



        plt.plot(np.arange(-200, 1000, 2), mean_erp)
        lowerbound = np.percentile(mean_erp[0:100, :].flatten(), 5)
        upperbound = np.percentile(mean_erp[0:100, :].flatten(), 95)
        plt.plot(np.arange(-200,0,2),np.matlib.repmat(lowerbound,100,1), 'k--')
        plt.plot(np.arange(-200,0,2), np.matlib.repmat(upperbound, 100, 1), 'k--')


            # plt.subplot(2, 3, int(run))
            # lowerbound = np.percentile(erp[0:100, :].flatten(),5)
            # upperbound = np.percentile(erp[0:100, :].flatten(), 95)
            # ax = plt.plot(np.arange(-200, 500, 2), erp)
            # plt.plot(np.arange(-200,0,2),np.matlib.repmat(lowerbound,100,1), 'k--')
            # plt.plot(np.arange(-200,0,2), np.matlib.repmat(upperbound, 100, 1), 'k--')

        plt.savefig('/home/jenny/ostwald-data/clean-eeg-converted/' + '/Figures/ERPs/ERP-sub%s.png' % sub[4:] )


# this function loads EEG data from each run
def get_erp(subID, run):
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

    all_stim = left_hcp + left_hcnp + left_lcp + right_lcnp\
               + right_hcp + right_hcnp + right_lcp + right_lcnp
    all_stim.sort()

    # index the latency
    tstim = latency[all_stim]


    # remove EOG and ECG
    data = np.transpose(data)

    data = np.delete(data, slice(30,32), axis = 1)

    #re-reference the data
    ref_mean = np.mean(data, axis = 1)
    ref_mean = np.transpose(np.tile(ref_mean, [data.shape[1],1 ]))
    data = data - ref_mean

    # construct a time x channel x trial matrix for each run for 6s
    samples = int(6*sr)
    channelnum = data.shape[1]
    trialnum = tstim.shape[0]
    trialdata = np.zeros((samples,channelnum, trialnum))

    # epoch the data to create single-trial segments for 5s
    for i in np.arange(trialnum):
        time = tstim[i]
        trialdata[:,:, i] = data[time-1000: time+2000,:]

    # baseline correction for each trial 200s pre-stimulus
    for i in range(0,trialnum):
        baseline_mean = np.mean(trialdata[(1000 - 100):1000,:, i], axis=0)
        baseline_mean = np.tile(np.mean(trialdata[(1000 - 100):1000,:, i], axis=0), [trialdata.shape[0]-900, 1])
        trialdata[(1000 - 100):, :, i] = trialdata[(1000 - 100):, :, i]  - baseline_mean

    # select the window from the 5s trialdata to look at ERP
    erp = np.mean(trialdata[(1000 - 100):1500, :, :], axis = 2)
    plt.plot(np.arange(-200, 1000, 2), erp)
    return erp

# this function gets the list of subject and how many runs per subject
def get_sub(path):
    list_subID = []
    dir_list = os.listdir(path)
    for files in dir_list:
        if files[:3] == 'sub':
            list_subID.append(files)
            list_subID.sort()
    subIDs = [i[0:7] for i in list_subID]
    subIDs_run = dict(Counter(subIDs))
    subIDs = np.unique(subIDs)
    return subIDs, subIDs_run


# this function obtain a list of the color scheme
def get_color(ax):
    colorscheme = []
    for i in range(0,len(ax)):
        linecolor = ax[i].get_color()
        colorscheme.append(linecolor)
    return colorscheme

# electrode_location
def get_chanlocs():
    chanlocdic = read_mat('/home/jenny/ostwald-data/code/' + 'pdm_erp_electrode_locations.mat')
    chanlocs =  chanlocdic['chanlocs']
    return chanlocs

if __name__ == "__main__":
	main()