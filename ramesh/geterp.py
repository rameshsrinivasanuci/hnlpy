from pymatreader import read_mat
import numpy
import matplotlib
import matplotlib.pyplot as plt 
from scipy import signal
import csv

#import data 
datadict = read_mat('task3/s195_ses2_task3_final.mat')
#This bit was for importing ica files.  we should just calculate internally
#ica = numpy.array(datadict['ica'])
#mix = numpy.array(datadict['mix'])
#sep = numpy.array(datadict['sep'])
#organize data
data = numpy.array(datadict['data'])
artifact = numpy.array(datadict['artifact'])
sr = numpy.array(datadict['sr'])
artifact0 = artifact.sum(axis = 0)
artifact1 = artifact.sum(axis = 1)
#identify goodtrials and good channels.  
goodtrials = numpy.squeeze(numpy.array(numpy.where(artifact0 < 20)))
goodchan = numpy.squeeze(numpy.array(numpy.where(artifact1 < 20)))
#average erp.  
erp = numpy.mean(data[:,:,goodtrials],axis = 2)
#make a lowpass filter
wp = 15/(sr/2)                                                                                         
ws = 20/(sr/2)                                                                                         
gpass = 3;                                                                                           
gstop = 20;                                                                                          
n,wn = signal.buttord(wp,ws,gpass,gstop);                                                            
b,a = signal.butter(n,wn,'low');
#lowpass filter the erp
erpfilt = signal.filtfilt(b,a,erp,axis = 0,padtype='odd')


# open up indices
with open('/data/pdmattention/EWMAV_task3/TrainingData_task3_indices.csv', 'r') as f:
	reader = csv.reader(f)

	# iterate through data to find desired subject
	# format of csv file is "sub, indices"
	for row in reader:
		if row[0] == 's195':
			beh_ind = row[1:] # gets the behavioral indices 

# changes the type of data from str to int
for ii in range(0, len(beh_ind)):
	beh_ind[ii] = int(beh_ind[ii])

# changes nparray into list type for easier manipulation 
# saved it as goodtrials1 incase you wanted to use np functions instead
goodtrials1 = goodtrials.tolist()

# creates a list of matching indices from behavioral data and eeg data
BehEeg_ind = list(set(beh_ind) & set(goodtrials))

