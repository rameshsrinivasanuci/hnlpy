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

	

def path_reconstuct(currentSub, path):
	# reconstructs data paths for subject 
	# /data/pdmattention/task#/s###_ses#_task#_final.mat
	# /data/pdmattention/task#/s###_ses#_task#_expinfo.mat

	BehPath = path + currentSub + "task" +str(taskNum) + "_expinfo.mat"
	EEGPath = path + currentSub + "task" +str(taskNum) + "_final.mat"

	return BehPath, EEGPath


def ReadFiles(currentSub, path):
	# if beh_files 
	#	index rts, conds, corrects
	# 	then call EWMAV
	# elif eeg_files 
	# 	index artifact
	# call EEG_Data

	# return BehInd, EEGInd 
	pass

def get_BehData():
	# organizes behavioral data
	pass

def get_EEGData():
	# organizes eeg data
	pass

def EWMAV():
	# finds a cutoff threshold for behavioral data
	pass

def find_overlapIndices():
	# gets the overlapping indices of behavioral and eeg data that are considered "good" trials
	pass

def extract_data():
	# extracts the data based on indices saved in "find_overlapIndices'
	pass


def writeCSV():
	# puts all the indices in a csv file
	pass



