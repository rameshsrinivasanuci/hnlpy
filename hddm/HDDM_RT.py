
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
L = 2
l = 0.01
s = .5


def get_files(goal_dir):
	data_files = os.listdir(goal_dir) # gets all files in current directory 

	# list of files
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
				check != 's214' and check != 's220' and check != 's225' and check != 's228' and check != 's234' and check != 's240' and check != 's213' and check != 's235'):		

							sub_files.append(current)
							sub_IDs.append(current[0:4])

	return all_subfiles, sub_IDs, sub_files, sub_IDs


def get_data(current_sub):
	# function loads each subject's file
	arrays = {}

	f = h5py.File(goal_dir + current_sub)
	for k, v in f.items():
		arrays[k] = np.array(v)

	f.close()

	# indexes into np.array to get rt, condition, and correct data
	rts = (arrays['rt']).astype(int)
	conditions = (arrays['condition']).astype(int)
	corrects = arrays['correct']


	return rts, conditions, corrects

def make_CSV(data, indices, current_sub, index, y):

	if debug == True:
		print('CWD: ', os.getcwd())

	if y == 1:
		filename = '/data/pdmattention/EWMAV_task3/TrainingData_task3.csv'
		filename1 = '/data/pdmattention/EWMAV_task3/TrainingData_task3_indices.csv'
	else:
		filename = '/data/pdmattention/EWMAV_task3/TrainingData_task3_allsubs.csv'
		filename1 = '/data/pdmattention/EWMAV_task3/TrainingData_task3_allsubs_indices.csv'
	# gets current subject index and makes a list of current subject's ID
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



	# creates CSV files
	if index == 0:
		with open(filename, 'w') as csvFile:
			wr = csv.writer(csvFile) # returns a writer object responsible for
							 # converting
			                 # the user's data into strings; we can use this
			                 # object to manipulate excel files

			wr.writerow(('subj_idx', 'stim', 'rt', 'response')) # writes the headers
			wr.writerows(export_data) # writes the data

		with open(filename1, 'w') as csvFile:
			wr = csv.writer(csvFile)
			wr.writerow(('subj_idx', 'indices'))
			wr.writerow(data1)
	

	else:
		with open(filename,'a') as csvFile:
			wr = csv.writer(csvFile)
			wr.writerows(export_data)

		with open(filename1, 'a') as csvFile:
			wr = csv.writer(csvFile)
			wr.writerow(data1)
	

def EWMAV(rts, cond, correct, current_sub):
	# EWMAV function sorts RTs from fastest to slowest and gets rid of RTs that are below a particular threshold 
	# the threshold is determined based on an algorithm created by Joachim Vanderchove 

	# indices 
	index0 = [range(360)] # original indices
	index1 = [] # indices of nan trials

	# lists for each data set
	# these lists are used to index out values in np.array 
	# because they are easier to manipulate
	rt_list = []
	cond_list = []
	correct_list = []


	#for loop which iterates through rt data and gets rid of data in 
	# nan trials for all data sets
	for ii in range(len(rts)):

		check_nan = rts[ii][0] #indexes value

		if check_nan > 0: # checks for nan trials; if greater than 0 append to valid data list
			rt_list.append(rts[ii][0].item()/1000)
			cond_list.append(cond[ii][0].item())
			correct_list.append(correct[ii][0].item())
		
		elif check_nan <= 0:
			index1.append(ii)


	index2 = np.argsort(rt_list)# indices of sorted rt_list
	rt_list.sort()

	# sorts condition & correct with new rt indices arrangement 
	cond_list = [cond_list[i] for i in index2]
	correct_list = [correct_list[i] for i in index2]

	cond_list2 = []
	for ii in cond_list:
		if (ii == 1) or (ii == 2) or (ii == 3):
			cond_list2.append(ii)
		elif ii == 4:
			cond_list2.append(1)
		elif ii == 5:
			cond_list2.append(2)
		elif ii == 6:
			cond_list2.append(3)

	cond_list = cond_list2

	# computation based on Joachim's code
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

	eng = ME.start_matlab()


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

	if debug == True:
		print('percentile: ', np.percentile(rt_list, 5))

	plt.xlim(rt_1[0]-.1, 1.4)
	plt.savefig('/data/pdmattention/EWMAV_task3/task3_RT_Figs/'+current_sub[0:5]+'EWMAVfig.png')


	if debug == True:
		plt.show(block=True)
		

	plt.close()
	
	data = [cond_list, rt_list, correct_list]

	return data, index2



all_subfiles, sub_IDs, sub_files, sub_IDs = get_files(goal_dir)

y1 = 1
y2 = 2

for xx in range(len(sub_files)):
	current_sub = sub_files[xx]
	rts, conditions, corrects = get_data(current_sub)
	data, index2 = EWMAV(rts, conditions, corrects, current_sub)
	make_CSV(data, index2, current_sub, xx, y1)

for xx in range(len(all_subfiles)):
	current_sub = all_subfiles[xx]
	rts, conditions, corrects = get_data(current_sub)
	data, index2 = EWMAV(rts, conditions, corrects, current_sub)
	make_CSV(data, index2, current_sub, xx, y2)
