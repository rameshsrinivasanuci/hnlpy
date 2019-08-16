# Mariel Tisby
# 8/15/19

'''
HNL Project

loads *.mat files into Python
'''

goal_dir = '/data/pdmattention/task3/'

import os
import h5py
import numpy as np


def load_data(goal_dir):
	data_files = os.listdir(goal_dir)

	print(data_files)

	sub_files = []
	sub_IDs = []

	all_subfiles = []
	all_subIDs = []

	# iterate through the goal directory and get all the data files 
	for current in data_files:
		if current[16:27] == 'expinfo.mat':
			all_subfiles.append(current)
			all_subIDs.append(current[0:4])

			check = current[0:4]
			if ( check != 's184' and check != 's187' and check != 's190' and check != 's193' and check != 's199' and check != 's209' and \
				check != 's214' and check != 's220' and check != 's225' and check != 's228' and check != 's234' and check != 's240' and check != 's213'):		

							sub_files.append(current)
							sub_IDs.append(current[0:4])


	print(all_subfiles)

	all_rt = []
	all_cond = []
	all_correct = [] 
	all_sub = []

	# iterate through each subject and get all their data
	for sub in range(len(sub_files)):
		current = sub_files[sub]
		#print('current: ', current, '\n')
		arrays = {}

		f = h5py.File(goal_dir + current)
		for k, v in f.items():
			arrays[k] = np.array(v)

		f.close()

		rts = (arrays['rt']).astype(int)
		conditions = (arrays['condition']).astype(int)
		corrects = arrays['correct']

		
		rt_list = []
		correct_list = []
		condition_list = []
		sub_list = [] # list of 1 sub's id 

		for ii in range(len(rts)):
			rt_list.append((rts[ii][0]/1000)) # change milliseconds to seconds -- seconds required for hddm to work 
			correct_list.append(corrects[ii][0])
			condition_list.append(conditions[ii][0])
			sub_list.append(current[0:4])

		print('sub: ', current, 'rt: ', rt_list)
		all_rt.append(rt_list)
		all_cond.append(condition_list)
		all_correct.append(correct_list)
		all_sub.append(sub_list)


	
	return [all_rt, all_cond, all_correct, all_sub]


