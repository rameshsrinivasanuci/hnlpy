# Mariel Tisby
# HNL Projects
# 5.28.19

'''
source activate snakes

'''

import h5py
import numpy as np

f = h5py.File('s181_ses1_task3_expinfo.mat')

array = {}
for k, v in f.items():
    arrays[k] = np.array(v)


# show dict_keys
arrays.keys()

rts = arrays['rt']

# check
length = len(rts)
add = sum(rts)
