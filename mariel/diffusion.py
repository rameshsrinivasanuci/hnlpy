# module for preprocessing data for hierarichal diffusion modeling

#imports
import os
import h5py
import numpy as np
import csv
import math
from scipy.signal import lfilter
import matlab.engine as ME
import matplotlib.pyplot as plt
from pymatreader import read_mat
from scipy import signal

debugging = True
showPlot = False

#global variables
L = 2
l = 0.01
s = .5
base_dir = "/data/pdmattention/"

def get_paths(path_type, taskNum):
	# this function specifies what path a function should follow
	# path_type = specifies if you want to get into the directory where data is stored or the 
	# directory where we want to store processed data
	# taskNum = the task we are analyzing

	if path_type == base_dir:
		path = base_dir + 'task'+ str(taskNum) + '/'

	else:  
		path = base_dir + "EWMAV_task" + str(taskNum) + '/'

	return path

def choose_subs(lvlAnalysis, path):
	# this function returns a list of subject IDs and task based on whether or not this is testing
	# analysis or analysis of all subjects
	# returns

	allDataFiles = os.listdir(path)

	if lvlAnalysis == 1:
		excludeSubs = ['s184', 's187', 's190', 's193', 's199', 's209', 's214', 's220', 's225', 's228', \
		's234', 's240', 's213', 's235']

		subIDs = [] 

		for files in allDataFiles:
			if (files[16:] == 'expinfo.mat') and files[0:4] not in excludeSubs == True:
				subIDs.append(files[0:10])

	elif lvlAnalysis == 2:
		for files in allDataFiles:
			if files[16:] == 'expinfo.mat':
				subIDs.append(files[0:10])

	return subIDs

	

def path_reconstuct(currentSub, path, taskNum):
	# reconstructs data paths for subject 
	# /data/pdmattention/task#/s###_ses#_task#_final.mat
	# /data/pdmattention/task#/s###_ses#_task#_expinfo.mat

	BehPath = path + currentSub + "task" +str(taskNum) + "_expinfo.mat"
	EEGPath = path + currentSub + "task" +str(taskNum) + "_final.mat"

	return BehPath, EEGPath

def ReadFiles(BehPath, EEGPath, taskNum, currentSub):
	behDict = read_mat(BehPath)
	rts = (behDict['rt']).tolist()
	rts = [x/1000 for x in rts]
	

	conds = (behDict['condition']).tolist()
	corrects = behDict['correct'].tolist()
	BehInd = EWMAV(rts, conds, corrects, taskNum, currentSub)

	EEGDict = read_mat(EEGPath)
	EEGInd = EEGData(EEGDict)

	return BehInd, EEGInd 

def EEGData(EEGDict):
	artifact = EEGDict['artifact']
	artifact0 = artifact.sum(axis = 0)
	goodtrials = np.squeeze(np.array(np.where(artifact0 < 20)))

	return goodtrials

def EWMAV(rts, conds, corrects, taskNum, currentSub):
	# finds a cutoff threshold for behavioral data and sorts RTs in descending order
	rt_list = []
	cond_list = []
	correct_list = []
	indices = []


	checkNaN = 0
	for xx in range(len(rts)):
		if rts[xx] > checkNaN:
			rt_list.append(rts[xx])
			cond_list.append(conds[xx])
			correct_list.append(corrects[xx])
			indices.append(xx)


	indices = (np.argsort(rt_list)).tolist() # gets the indices of RTs in ascending order
	rt_list.sort()

	cond_list = [cond_list[i] for i in indices]
	correct_list = [correct_list[i] for i in indices]

	cond_list = rearrangeCond(cond_list, taskNum)

	# computation of EWMAV model based on Joachim's MATLAB code
	# these steps have to be broken down in python due to the use of
	# different toolboxes
	step1 = np.power((1-l), np.multiply(2,(range(len(rt_list)))))
	step1 = np.round(step1,4)
	step2 = np.round((l/(2-l)), 4)
	step4 = 1 - step1
	step3 = np.round(step4*step2,4)
	step5 = np.round(np.sqrt(step3), 4)

	ucl = .5+L*s*step5
	lcl = .5-L*s*step5

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
	indices2 = []
	for ii in range(len(rt_list)):
		if z1[ii] > ucl[ii]:
			rt_1.append(rt_list[ii])
			indices2.append(indices[ii])
			z_1.append(z[ii])
		else:
			rt_2.append(rt_list[ii])
			z_2.append(z[ii])

	percentile = np.percentile(rt_list,5).item()

	rt_list2 = []
	for ii in rt_list:
		rt_list2.append(round(ii, 2))


	xx = rt_list2.index(round(percentile, 2))

	'''
	plt.plot(rt_1, z_1, 'b.')
	plt.plot(rt_2, z_2, 'r.')
	plt.plot(rt_list, ucl, 'm:')
	plt.plot(rt_list2[xx], z[xx], 'yv')

	currentSub = currentSub[0:5]

	plt.xlim(rt_1[0]-.1, 1.4)
	plt.savefig('/data/pdmattention/EWMAV_task3/task3_RT_Figs'+currentSub+'EWMAVfig_L1_5.png')

	rt_list = rt_1
	'''

	return indices2


def rearrangeCond(cond_list, taskNum):
	condListCorrected = []

	if taskNum == 3:
		for ii in cond_list:
			if (ii == 1) or (ii == 2) or (ii == 3):
				condListCorrected.append(ii)
			elif ii == 4:
				condListCorrected.append(1)
			elif ii == 5:
				condListCorrected.append(2)
			elif ii == 6:
				condListCorrected.append(3)
	else:
		pass


	return condListCorrected

def find_overlapIndices(BehInd, EEGInd):
	# gets the overlapping indices of behavioral and eeg data that are considered "good" trials
	BehEeg_ind = (set(BehInd) & set(EEGInd))
	return BehEeg_ind

def extract_data(OverlapInd, currentSub, path, taskNum):
	# extracts the data based on indices saved in "find_overlapIndices'
	# reconstrucct path
	dataPath = path + currentSub[0:4] + '_behavior_final.mat'
	dataDict = read_mat(dataPath)

	rt_list = []
	correct_list = []
	condition_list = []
	eeg_list = []
	sub_list = []

	rts = (dataDict['rt']).tolist()
	corrects = (dataDict['correct']).tolist()
	conditions = (dataDict['condition']).tolist()
	eeg = (dataDict['trial']).tolist()

	for index in OverlapInd:
		rt_list.append(float(rts[index.astype(int)]))
		correct_list.append(float(corrects[index.astype(int)]))
		condition_list.append(int(conditions[index.astype(int)]))
		eeg_list.append(eeg[index])
		sub_list.append(currentSub[0:4])

	
	data = [sub_list, condition_list, rt_list, correct_list, eeg_list]
	if debugging == True:
		print(type(eeg_list))
	
	return data


def writeCSV(sub_data, iteration, taskNum, lvlAnalysis):
	# puts all the indices in a csv file
	if lvlAnalysis == 1:
		lvlAnalysis = 'testing'
	else:
		lvlAnalysis = 'allSubs'

	path_type = 'store_data'
	StorePath = get_paths(path_type, taskNum)
	filename = StorePath + 'TrainingData_task' + str(taskNum) + lvlAnalysis

	if iteration == 0:
		with open(filename, 'w') as csvFile:
			wr = csv.writer(csvFile) # returns a writer object responsible for
							 # converting
			                 # the user's data into strings; we can use this
			                 # object to manipulate excel files

			wr.writerow(('subj_idx', 'stim', 'rt', 'response', 'artifact')) # writes the headers
			wr.writerows(sub_data) # writes the data

	else:
		with open(filename,'a') as csvFile:
			wr = csv.writer(csvFile)
			wr.writerows(data)



def writeCSVColumns(data, filenmae, newfilename):
	with open (filename, 'r') as readobj, open(newfilename, 'w', newline='') as wrtieobj:
		csv_reader = reader(read_obj)
		csv_writer = writer(write_obj)

		index = 0
		for row in csv_reader:
			reow.append(data[index])
			csv_writer.writerows(row)
			index = index + 1 