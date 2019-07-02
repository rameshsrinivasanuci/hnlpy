# Mariel Tisby
# HNL Project CSV Conversion Multiple Subject Data Files
# 6.27.19

import h5py 
import numpy as np 
import os
import csv
from itertools import zip_longest 

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
sub_files = []
sub_IDs = []

for cur_file in data_files: # iterates through every file in directory
	# if os.path.isfile(cur_file) == True: # checks to see if file is a file
		if cur_file[16:27] == 'expinfo.mat': # specifies the file type
			sub_files.append(cur_file)
			sub_IDs.append(cur_file[0:4])

# goes through every subject and extracts data 
for sub in range(len(sub_files)):
	
	current_sub = sub_files[sub]
	print('subject:',current_sub)
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

	rts = arrays['rt']
	conditions = arrays['condition']
	corrects = arrays['correct']

	# extracts all data and puts data in a list
	for ii in range(len(rts)):
		rt_list.append(rts[ii][0])
		correct_list.append(corrects[ii][0])
		condition_list.append(conditions[ii][0])
		sub_list.append(sub_IDs[sub])

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

	
