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

# subID = 'sub-01'
# run = '01'
basedir = '/home/jenny/motion-data/'
path = '/home/jenny/motion-data/'

# enter subID as sub-##, run as 01 or 02
def main():
    getall_erp()

# this function cycles through all subjects and generate figures for each run
def getall_erp():
    subIDs = []
    dir_list = os.listdir(path)
    for files in dir_list:
        if files[:3] == 'sub':
            subIDs.append(files)
            subIDs.sort()

    for sub in subIDs:
        plt.figure(figsize=(14,14))

        erp1, erpfiltproject1 = get_erp(sub,'01')
        lowerbound = np.percentile(erp1[1800:2000, :].flatten(),5)
        upperbound = np.percentile(erp1[1800:2000, :].flatten(), 95)

        erp2, erpfiltproject2 = get_erp(sub, '02')
        lowerbound = np.percentile(erp2[1800:2000, :].flatten(), 5)
        upperbound = np.percentile(erp2[1800:2000, :].flatten(), 95)

        plt.subplot(221)
        plt.plot(np.arange(-200,500), erp1[1800:2500, :])
        plt.plot(np.arange(-200,0),np.matlib.repmat(lowerbound,200,1), 'k--')
        plt.plot(np.arange(-200,0), np.matlib.repmat(upperbound, 200, 1), 'k--')

        plt.subplot(222)
        plt.plot(np.arange(-200,500), erp2[1800:2500, :])
        plt.plot(np.arange(-200, 0), np.matlib.repmat(lowerbound, 200, 1), 'k--')
        plt.plot(np.arange(-200, 0), np.matlib.repmat(upperbound, 200, 1), 'k--')
        plt.title('ERP for subject %s' % sub[4:])

        plt.subplot(223)
        plt.plot(np.arange(-200, 500), erpfiltproject1[1800:2500])
        plt.title('single estimate ERP for subject %s' % sub[4:])

        plt.subplot(224)
        plt.plot(np.arange(-200,500), erpfiltproject2[1800:2500])
        plt.title('single estimate ERP for subject %s' % sub[4:])

        plt.savefig(path + '/Figures/ERPs/ERP-sub%s.png' % sub[4:] )

# load the data and calculate the ERPs
def  get_erp(subID, run):
    currentSub = subID
    currentRun = 'run-'+ run
    print('Current Subject: ', currentSub)
    print('Current Run:', currentRun)
    datadict = read_mat(path + subID + '/EEG/' + 'EEG_data_'+ subID + '_' + currentRun + '.mat')
    eventsdict = read_mat(path + subID + '/EEG/' + 'EEG_events_' + subID + '_' + currentRun + '.mat')

    data = np.array(datadict['EEGdata']['Y'])
    sr = np.array(datadict['fs'])
    tresp = np.array(eventsdict['tresp'])
    tstim = np.array(eventsdict['tstim'])

    # construct a time x channel x trial matrix for each run
    channelnum = data.shape[0]
    trialnum = tresp.shape[0]
    trialdata = np.zeros((5000,channelnum, trialnum))
    data = np.transpose(data)
    for i in np.arange(trialnum):
        time = tstim[i]
        trialdata[:,:, i] = data[time-2000: time+3000,:]

    erp = np.mean(trialdata[:,:,:],axis = 2)

    # make a lowpass filter
    sos, w, h = timeop.makefiltersos(sr, 10, 20)
    erpfilt = signal.sosfiltfilt(sos, erp, axis=0, padtype='odd')
    erpfiltbase = timeop.baselinecorrect(erpfilt, np.arange(1849,1998,1))

    # Identify an optimal set of weights to estimate a single erp peak.
    u,s,vh = linalg.svd(erpfiltbase[2150:2375,:])
    weights = np.zeros((channelnum,1))
    weights[:,0] = np.matrix.transpose(vh[0,:])

    erpfiltproject = np.matmul(erpfiltbase,weights)
    erpmin = np.amin(erpfiltproject[2150:2375])
    erpmax = np.amax(erpfiltproject[2150:2375])
    if abs(erpmin) < abs(erpmax):
        weights = -weights
        erpfiltproject = -erpfiltproject

    erp_peaktiming = np.argmin(erpfiltproject[2150:2375]) + 2150
    indices = np.arange(erp_peaktiming - 10, erp_peaktiming + 10, 1)
    erp_peakvalue = np.mean(erpfiltproject[indices])
    return erp, erpfiltproject

def


# to plot for individual subjects
# plt.plot(np.arange(-200,500), erp[1800:2500, :])
# plt.plot(np.arange(-200,500), erpfiltproject[1800:2500])

if __name__ == "__main__":
	main()