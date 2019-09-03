# Mariel Tisby
# 8/15/19

'''
HNL Project

This project is the main function for sorting through HDDM rt data 
includes:
	~ CSV file outputs with subject data
	~ MAT file output with indices of removed trials and original trial order
	~ Removal of NAN trials
	~ cutoff function using EWMAV

'''
# Debuging
debug = False

# imports
import os
import h5py
import numpy as np
import csv
import math
from itertools import zip_longest 
from scipy.signal import lfilter
import matlab.engine as ME
import matplotlib.pyplot as plt

# Global Variables
goal_dir = '/data/pdmattention/task3/'
L = 1.5
l = 0.01
s = .5


def get_files(goal_dir):
	data_files = os.listdir(goal_dir)

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

	return all_subfiles, sub_IDs, sub_files, sub_IDs


def get_data(current_sub):
	arrays = {}

	f = h5py.File(goal_dir + current_sub)
	for k, v in f.items():
		arrays[k] = np.array(v)

	f.close()

	rts = (arrays['rt']).astype(int)
	conditions = (arrays['condition']).astype(int)
	corrects = arrays['correct']


	return rts, conditions, corrects

def make_CSV(data, indices, current_sub, index):

	sub = current_sub[0:4]
	sub_list = []
	for xx in range(len(data[0])):
		sub_list.append(sub)

	data.insert(0, sub_list)
	data1 = [sub, indices]

	export_data = zip_longest(*data, fillvalue='') # zip_longest makes an iterator 
		                                       # that aggregates elements from
		                                       # each of the iterables; if uneven
		                                       # missing values are filled-in
		                                       # with fillvalue


	if index == 0:
		with open('TestData.csv', 'w') as csvFile:
			wr = csv.writer(csvFile) # returns a writer object responsible for
							 # converting
			                 # the user's data into strings; we can use this
			                 # object to manipulate excel files

			wr.writerow(('subj_idx', 'condition', 'rt', 'correct')) # writes the headers
			wr.writerows(export_data) # writes the data

		with open('HDDM_Indices.csv', 'w') as csvFile:
			wr = csv.writer(csvFile)
			wr.writerow(('subj_idx', 'indices'))
			wr.writerow(data1)
	

	else:
		with open('TestData.csv','a') as csvFile:
			wr = csv.writer(csvFile)
			wr.writerows(export_data)

		with open('HDDM_Indices.csv', 'a') as csvFile:
			wr = csv.writer(csvFile)
			wr.writerow(data1)
	

def EWMAV(rts, cond, correct):
	# indices 
	index0 = [range(360)]
	index1 = []

	# lists 
	rt_list = []
	cond_list = []
	correct_list = []

	for ii in range(len(rts)):

		check_nan = rts[ii][0]

		if check_nan > 0: # checks for nan trials 
			rt_list.append(rts[ii][0].item()/1000)
			cond_list.append(cond[ii][0].item())
			correct_list.append(correct[ii][0].item())
		
		elif check_nan <= 0:
			index1.append(ii)


	# indices of sorted rt_list
	index2 = np.argsort(rt_list) 
	rt_list.sort()

	# sorts condition & correct with new rt indices arrangement 
	cond_list = [cond_list[i] for i in index2]
	correct_list = [correct_list[i] for i in index2]

	# computation
	check1 = np.power((1-l), np.multiply(2,(range(len(rt_list)))))
	check1 = np.round(check1,4)
	check4 = 1-check1
	check2 = np.round((l/(2-l)), 4)
	check3 = np.round(check4*check2,4)
	check5 = np.round(np.sqrt(check3), 4)

	ucl = .5+L*s*check5
	lcl = .5-L*s*check5


	b = np.array([l])
	a = np.array([1, (l-1)])
	x = np.array(correct_list)
	zi = np.array((1-l)/2)

	z = []
	for n in range(len(x)):
		
		if n == 0:
			z_add = (b[0]) * x[n]- (a[1]) * zi
			z_add = a[0]*z_add
			z.append(round(z_add, 4)) 
		else:
			z_add = (b[0]) * x[n] - (a[1]) * z[n-1]
			z_add = a[0]*z_add
			z.append(round(z_add, 4)) 


	z1 = np.array(z)

	ucl1 = []

	for ii in range(len(ucl)):
		ucl1.append(ii)

	#vv = np.argwhere((np.diff(z<ucl1)== -1), 1, 'first')
	eng = ME.start_matlab()

	# vvv have to break into two lines of code because np.diff returns bool instead of 1s and 0s vvv
	# vv = eng.find(np.diff(z1<ucl)==-1, 1, 'first')

	Diff = ((np.diff(z1<ucl) == -1).astype(int))

	if np.count_nonzero(Diff)==0:
		vv = []
	else:
		vv1 = np.where(Diff>0)
		vv = vv1[0].item()

	eps = 2.2204e-16

	cutoff = 0

	if eng.isempty(vv) == False:
		cutoff = rt_list[vv] + eps
	else:
		vv = 1

	rt_1 = []
	rt_2 = []
	z_1 = []
	z_2 = []

	slope1_list = []
	slope2_list = []
	for ii in range(len(rt_list)):
		if z1[ii] > ucl[ii]:
			rt_1.append(rt_list[ii])
			z_1.append(z[ii])
		else:
			rt_2.append(rt_list[ii])
			z_2.append(z[ii])

	percentile = np.percentile(rt_list,5).item()

	rt_list2 = []
	for ii in rt_list:
		rt_list2.append(round(ii, 2))


	xx = rt_list2.index(round(percentile, 2))

	plt.plot(rt_1, z_1, 'b.')
	plt.plot(rt_2, z_2, 'r.')
	plt.plot(rt_list, ucl, 'm:')
	plt.plot(rt_list2[xx], z[xx], 'yv')
	print('percentile: ', np.percentile(rt_list, 5))


	plt.xlim(rt_1[0]-.1, 1.4)
	plt.show(block=True)

	sub_list = []

	
	data = [cond_list, rt_list, correct_list]

	return data, index2



all_subfiles, sub_IDs, sub_files, sub_IDs = get_files(goal_dir)

for xx in range(len(sub_files)):
	current_sub = sub_files[xx]
	rts, conditions, corrects = get_data(current_sub)
	data, index2 = EWMAV(rts, conditions, corrects)
	make_CSV(data, index2, current_sub, xx)

