# Mariel Tisby
# HNL Projects
# 5.28.19

'''
source activate snakes

'''

import h5py
import numpy as np
import pandas as pd
import csv

f = h5py.File('s181_ses1_task3_expinfo.mat')

arrays = {}
for k, v in f.items():
    arrays[k] = np.array(v)


# show dict_keys
arrays.keys()

rts = (arrays['rt']).tolist()
corrects = (arrays['correct']).tolist()
conditions = (arrays['condition']).tolist()

variables = [rts, corrects, conditions]
iters = len(variables)

rt_list = []
correct_list = []
condition_list = []

for kk in iters:
    variable = variables[kk]
    for jj in variable:
        if kk == 0:
            rt_list.append(jj[0])
        elif kk == 1:
            correct_list.append(jj[0])
        else:
            condition_list.append(jj[0])

var_dict = {'rt':rt_list,
            'correct':correct_list,
            'condition':condition_list}

pd.Series(var_dict)

# can take matlab dict and convert to pandas data structure
# pd.Series(arrays) BUT...

# we only want
# stim -- easy, medium, hard
# 1 & 4
# 2 & 5
# 3 & 6
# separation in spatial freq
# correct -- response



# figure out the array conversion


'''
for i in arrays.keys():
    data = arrays[i]

    if i == 'rt':

    elif i == 'correct':

    elif i == 'condition':
'''


# see if you can take everything out of the narray func into a normal list
# then use the dict function in pd.Series
