
from pymatreader import read_mat
import numpy
import matplotlib
import matplotlib.pyplot as plt 
from scipy import signal
import csv
import os

goal_dir = '/data/pdmattention/task3/'

files = os.listdir('/data/pdmattention/task3/')

subjects = []
beh_inds = []
with open('/data/pdmattention/EWMAV_task3/TrainingData_task3_indices.csv', 'r') as f:
	reader = csv.reader(f)

	count = 0
	for row in reader:
		if count != 0:
			subject = row[0]
			beh_ind = row[1:]

			subjects.append(subject)
			beh_inds.append([beh_ind])
		count = count + 1


# iterate through data to find desired subject
# format of csv file is "sub, indices"
count = 0
for item in subjects:
	subject = item
	beh_ind = beh_inds[count][0]

	for file in files:
		if file.endswith('final.mat') == True and file.startswith(str(subject)) == True:
			# print("are you doing anything")
			final_file = file

	datadict = read_mat(goal_dir+final_file)
	artifact = numpy.array(datadict['artifact'])

	artifact0 = artifact.sum(axis = 0)
	artifact1 = artifact.sum(axis = 1)
	#identify goodtrials and good channels.  
	goodtrials = numpy.squeeze(numpy.array(numpy.where(artifact0 < 20)))
	goodchan = numpy.squeeze(numpy.array(numpy.where(artifact1 < 20)))
	
	for x in range(len(beh_ind)):
		beh_ind[x] = int(beh_ind[x])

	goodtrials1 = goodtrials.tolist()

	# creates a list of matching indices from behavioral data and eeg data
	BehEEG_ind = list(set(beh_ind) & set(goodtrials))
	BehEEG_ind.insert(0, subject)

	if count == 0:
		with open('/data/pdmattention/EWMAV_task3/TrainingData_task3_FinalIndices.csv', 'w') as csvFile:
			wr = csv.writer(csvFile)
			wr.writerow(('subj_idx', 'indices'))
			wr.writerow(BehEEG_ind)
	else:
		with open('/data/pdmattention/EWMAV_task3/TrainingData_task3_FinalIndices.csv', 'a') as csvFile:
			wr = csv.writer(csvFile)
			wr.writerow(BehEEG_ind)

	count = count + 1


