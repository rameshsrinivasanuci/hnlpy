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

	if path _type == 1:
		path = base_dir + "task" + str(task) + "/"
	else:
		path = base_dir + "EWMAV_" + "task" + str(task) + "/"


	return path

def get_files(path):
	# this function gets the full path names for all data
	# this function should be used only be BehData
	
	data_files = os.listdir(path)
	sub_files = []
	sub_IDs = []

	all_subfiles = []
	all_subIDs = []

	for currentFile in data_files:
		if currentFile[16:27] == "expinfo.mat":
			all_subfiles.append(currentFile)
			all_subIDs.append(currentFile[0:4])

			checkFile = currentFile[0:4]
			if ( checkFile != 's184' and checkFile != 's187' and checkFile != 's190' and checkFile != 's193' and checkFile != 's199' and checkFile != 's209' and \
				checkFile != 's214' and checkFile != 's220' and checkFile != 's225' and checkFile != 's228' and checkFile != 's234' and checkFile != 's240' and checkFile != 's213' and checkFile != 's235'):		
							sub_files.append(currentFile)
							sub_IDs.append(currentFile[0:4])

	return all_subfiles, all_subIDs, sub_files, sub_IDs


def get_BehData():
	# organizes behavioral data
	pass

def get_EEGData():
	# organizes eeg data
	pass

def EWMAV():
	# finds a cutoff threshold for behavioral data
	

def create_indices():
	# puts all the indices in a csv file
	pass

def find_overlapIndices():
	# gets the overlapping indices of behavioral and eeg data that are considered "good" trials
	pass

def extract_data():
	# extracts the data based on indices saved in "find_overlapIndices"
	pass

def PickAnalysisLvl(lvlAnalysis, allFiles, allIDs, testFiles, testIDs):
	if lvlAnalysis == 1:
		files = testFiles
		IDs = testIDs
	else:
		files = allFiles
		IDs = allIDs

	return files, IDs

def ReadFiles(currentSub, path):
	datadict = read_mat(path + currentSub)
	rts = datadict['rt']
	conditions = datadict['condition']
	corrects = datadict['corrects']

	return rts, conditions, corrects