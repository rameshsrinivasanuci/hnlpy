


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
from pymatreader import read_mat
from hdf5storage import savemat
from hdf5storage import loadmat
results = loadmat('/home/jenny/Downloads/behavior2_task3_HDDM_AlphaJan_20_21_14_04_estimates.mat')
delta = np.mean(results['alpha']['mean'][0][0], axis=1)
subj = loadmat('/home/jenny/Downloads/behavior2_task3')['uniquepart'][0]
len(subj['uniquepart'][0])
path = '/home/jenny/pdmattention/'
subIDs = choose_subs(1, path + 'task3')
subIDs.remove('s236_ses1_')

subID2 = [i[1:4] for i in subIDs]
subj_sort = subj.searchsorted(subID2 )
newdelta = delta[subj_sort]

# now link

# this is the code to load subjects
def loaddata(electrode):
    electrode = electrode[4:]
    subject_bycond = np.load(path + 'alphaDC/'+'subject_bycond_'+electrode+'.npy')
    subject_rt = np.load(path + 'alphaDC/'+'subject_rt_' + electrode+'.npy')
    subject_acc = np.load(path + 'alphaDC/'+'subject_acc_' + electrode+'.npy')
    subject_accrt = np.load(path + 'alphaDC/'+'subject_accrt_'+electrode+'.npy')
    return subject_bycond, subject_rt, subject_acc, subject_accrt

subject_bycond, subject_rt, subject_acc, subject_accrt = loaddata('out.allchans')

# test one subject
data = loadmat('/home/jenny/pdmattention/task3/final_interp/s231_ses2_final_interp.mat')['data']
data_erp = np.mean(data, axis= 2)