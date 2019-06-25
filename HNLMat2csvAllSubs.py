# Mariel Tisby
# HNL Projects
# 5.28.19


# This project intends to open up all subject data from .mat files and
# convert them into csv files 

'''
source activate snakes

'''

import h5py
import numpy as np
import pandas as pd
import csv
from itertools import zip_longest
from pathlib import Path

p = Path("/data/pdmattetion/task3")
sub_files = []
subIDs = []
for file in p.iterdir():
    if file.is_file() == True:
        if file.parts[4][16:27] == 'expinfo.mat':
            sub_files.append(file)
            subIDs.append(file.parts[4][:4])
        

for file_num in range(len(sub_files)):
    openfile = h5py.File(sub_files[file_num])

    arrays = {}
    for k, v in openfile.items():
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
    get = len(rt_list)

    subidx = []
    for xx in range(get):
        subidx.append(subIDs[file_num])

    data = [subidx, condition_list, rt_list, correct_list]

    # the following lines takes our data and writes it into a csv file

    export_data = zip_longest(*data, fillvalue='') # zip_longest makes an iterator 
                                                   # that aggregates elements from
                                                   # each of the iterables; if uneven
                                                   # missing values are filled-in
                                                   # with fillvalue

    if file_num == 0:                                               
        with open('prac_w_csv.csv', 'w', encoding = 'ISO-8859_1', newline='') as csvFile:
            wr = csv.writer(csvFile) # returns a writer object responsible for converting
                                     # the user's data into strings; we can use this
                                     # object to manipulate excel files

            # change order to match cavanagh data 
            wr.writerow('subj_idx', 'condition', 'rt', 'correct')) # writes the headers
            wr.writerows(export_data) # writes the data

        csvFile.close()
    else:
        with open('prac_w_csv.csv', 'a', encoding = 'ISO-8859_1', newline='') as csvFile:
            wr = csv.writer(csvFile)
            wr.writerows(export_data)
        csvFile.close()


# we only want
# stim -- easy, medium, hard
# 1 & 4
# 2 & 5
# 3 & 6
# separation in spatial freq
# correct -- response


#sublime

# run simple_model.py
# terminator

