import h5py 
import numpy as np 
import os

# get // check current directory
cwd = os.getcwd()
print('Current Directory: ', cwd)

# change directory
new_dir = '/data/pdmattention/task3' #type file path for new directory with data
os.chdir(new_dir)
print('Current Directory: ', os.getcwd())

# get the files in the target directory
# file names are stored in variable sub_files 
# subject ids are stored in variable sub_IDs
data_files = os.listdir(cwd)

sub_files = []
sub_IDs = []

for cur_file in data_files: # iterates through every file in directory
	if os.path.isfile(cur_file) == True: # checks to see if file is a file
		if cur_file[16:27] == 'expinfo.mat': # specifies the file type
			sub_files.append(cur_file)
			sub_IDs.append(cur_file[0:4])


# test if function works -- index one subject and 
# open file using h5py
# must import numpy due to new Matlab update v7.3
sub = sub_files[0]
arrays = {}

f = h5py.File(sub)
for k, v in f.items():
	arrays[k] = np.array(v)

f.close()







