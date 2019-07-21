# Mariel Tisby
# HNL Project CSV Conversion Multiple Subject Data Files
# 6.27.19

# 7/3/19 updated to create two CSV files ~ one with all subjets and one without 12 subjects 
# 7/3/19 updated to change rts from msec to sec because hddm works only for seconds 

import h5py 
import numpy as np 
import os
import csv
from itertools import zip_longest 
import math
import SortRts

# get // check current directory
cwd = os.getcwd()
print('Current Directory: ', cwd)

# change directory
new_dir = '/data/pdmattention/task3/' #type file path for new directory with data
save_dir = '/data/pdmattention/'
# os.chdir(new_dir)
print('Current Directory: ', os.getcwd())

# get the files in the target directory
# file names are stored in variable sub_files 
# subject ids are stored in variable sub_IDs
data_files = os.listdir(new_dir)
sub_files = [] # list of subject files excluding 12 subjects
sub_IDs = [] # list of subject IDs excluding the same 12 subjects as above

all_subfiles = [] # list of all subject files
all_subIDs = [] # list of all subject IDs

for cur_file in data_files: # iterates through every file in directory
	# if os.path.isfile(cur_file) == True: # checks to see if file is a file
		if cur_file[16:27] == 'expinfo.mat': # specifies the file type
			all_subfiles.append(cur_file)
			all_subIDs.append(cur_file[0:4])

			check = cur_file[0:4]
			if ( check != 's184' and check != 's187' and check != 's190' and check != 's193' and check != 's199' and check != 's209' and \
				check != 's214' and check != 's220' and check != 's225' and check != 's228' and check != 's234' and check != 's240' and check != 's213'):

				sub_files.append(cur_file)
				sub_IDs.append(cur_file[0:4])

nantrial_dict_exsub = dict()

# goes through every subject and extracts data and writes to a 
# csv file excluding the 12 subjects --> write to file TestData.csv
for sub in range(len(sub_files)):
	
	current_sub = sub_files[sub]
	
	arrays = {}

	# opens file and extracts all data using numpy 
	f = h5py.File(new_dir + current_sub)
	for k, v in f.items():
		arrays[k] = np.array(v)

	f.close()

	rt_list = []
	correct_list = []
	condition_list = []
	sub_list = []
	indices1 = []

	rts = arrays['rt']
	conditions = arrays['condition']
	corrects = arrays['correct']

	# call SortRTs

	# extracts all data and puts data in a list
	for ii in range(len(rts)):
		
		check_nan = math.isnan(rts[ii][0])
		
		if check_nan == False: # checks for nan trials 
			rt_list.append((rts[ii][0]/1000)) # change milliseconds to seconds -- seconds required for hddm to work 
			correct_list.append(corrects[ii][0])
			condition_list.append(conditions[ii][0])
			sub_list.append(sub_IDs[sub])
		elif check_nan == True:
			indices1.append(ii) # makes a list of the index of nan trials per sub


	# puts nan indices in dictionary
	nantrial_dict_exsub[current_sub[0:4]] = indices1 

	data = [sub_list, condition_list, rt_list, correct_list]

	# zips lists together, making it easier to write to csv file
	export_data = zip_longest(*data, fillvalue='') # zip_longest makes an iterator 
		                                       # that aggregates elements from
		                                       # each of the iterables; if uneven
		                                       # missing values are filled-in
		                                       # with fillvalue

	# change to directory you want csv file to be saved in 
	#os.chdir(save_dir)
	# os.chdir(cwd)
	
	if sub == 0:
		with open('/data/pdmattention/TestData.csv', 'w', encoding = 'ISO-8859_1', newline='') as csvFile:
			wr = csv.writer(csvFile) # returns a writer object responsible for
						 # converting
		                             # the user's data into strings; we can use this
		                             # object to manipulate excel files

			wr.writerow(('subj_idx', 'condition', 'rt', 'correct')) # writes the  											# headers
			wr.writerows(export_data) # writes the data

        		
	else:
		with open('/data/pdmattention/TestData.csv','a', encoding = 'ISO-8859_1', newline='') as csvFile:
			wr = csv.writer(csvFile)
			wr.writerows(export_data)
        		
		
	# os.chdir(new_dir) # change directory back to where data is

'''
# second for loop to create a file with all subject data
# written to TestDataAllSubs.csv
	
nantrial_dict_allsub = dict()

for sub in range(len(all_subfiles)):
	
	current_sub = all_subfiles[sub]

	arrays = {}

	# opens file and extracts all data using numpy 
	f = h5py.File(new_dir + current_sub)
	for k, v in f.items():
		arrays[k] = np.array(v)

	f.close()

	rt_list = []
	correct_list = []
	condition_list = []
	sub_list = []
	indices1 = []

	rts = arrays['rt']
	conditions = arrays['condition']
	corrects = arrays['correct']

	# extracts all data and puts data in a list
	for ii in range(len(rts)):
		
		check_nan = math.isnan(rts[ii][0]) # checks if rt value is nan
	
		if check_nan == False: # if real number append that index's rt, correct, and condition
			rt_list.append((rts[ii][0]/1000)) # change milliseconds to seconds -- seconds required for hddm to work 
			correct_list.append(corrects[ii][0])
			condition_list.append(conditions[ii][0])
			sub_list.append(all_subIDs[sub])
		elif check_nan == True:
			indices1.append(ii) # makes a list of the index of nan trials per sub


	# puts nan indices in dictionary
	nantrial_dict_allsub[current_sub[0:4]] = indices1 

	data = [sub_list, condition_list, rt_list, correct_list]


	# zips lists together, making it easier to write to csv file
	export_data = zip_longest(*data, fillvalue='') # zip_longest makes an iterator 
		                                       # that aggregates elements from
		                                       # each of the iterables; if uneven
		                                       # missing values are filled-in
		                                       # with fillvalue

	# change to directory you want csv file to be saved in 
	#os.chdir(save_dir)
	# os.chdir(cwd)
	
	if sub == 0:
		with open('/data/pdmattention/TestDataAllSubs.csv', 'w', encoding = 'ISO-8859_1', newline='') as csvFile:
			wr = csv.writer(csvFile) # returns a writer object responsible for
						 # converting
		                             # the user's data into strings; we can use this
		                             # object to manipulate excel files

			wr.writerow(('subj_idx', 'condition', 'rt', 'correct')) # writes the  											# headers
			wr.writerows(export_data) # writes the data

        		
	else:
		with open('/data/pdmattention/TestDataAllSubs.csv','a', encoding = 'ISO-8859_1', newline='') as csvFile:
			wr = csv.writer(csvFile)
			wr.writerows(export_data)

'''


with open('/data/pdmattention/NaNIndices_ExcludeSub.csv', 'w') as f:
	fieldnames = ['subj_id', 'indices']
	writer = csv.DictWriter(f, fieldnames = fieldnames)
	writer.writeheader()
	data = [dict(zip(fieldnames, [k,v])) for k, v in nantrial_dict_exsub.items()]
	writer.writerows(data)

'''
with open('/data/pdmattention/NaNIndices_AllSub.csv', 'w') as f:
	fieldnames = ['subj_id', 'indices']
	writer = csv.DictWriter(f, fieldnames = fieldnames)
	writer.writeheader()
	data = [dict(zip(fieldnames, [k,v])) for k, v in nantrial_dict_allsub.items()]
	writer.writerows(data)

'''
	