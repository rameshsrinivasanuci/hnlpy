from pymatreader import read_mat
import csv
import numpy
import scipy.signal as signal
import matplotlib
import matplotlib.pyplot as plt 
import timeop
from sklearn import decomposition

#import data 
datadict = read_mat('/home/ramesh/pdmattention/task3/s206_ses1_task3_final.mat')
#This bit was for importing ica files.  we should just calculate internally
#ica = numpy.array(datadict['ica'])
#mix = numpy.array(datadict['mix'])
#sep = numpy.array(datadict['sep'])
#organize data
data = numpy.array(datadict['data'])
artifact = numpy.array(datadict['artifact'])
sr = numpy.array(datadict['sr'])
# open up indices
with open('/home/ramesh/pdmattention/EWMAV_task3/TrainingData_task3_FinalIndices.csv', 'r') as f:
	reader = csv.reader(f)

	# iterate through data to find desired subject
	# format of csv file is "sub, indices"
	for row in reader:
		if row[0] == 's206':
			beh_ind = row[1:] # gets the behavioral indices 

# changes the type of data from str to int
for ii in range(0, len(beh_ind)):
	beh_ind[ii] = int(beh_ind[ii])


#average erp.  
erp = numpy.mean(data[:,:,beh_ind],axis = 2)
#make a lowpass filter
sos,w,h = timeop.makefiltersos(sr,15,20)
erpfilt = signal.sosfiltfilt(sos,erp,axis = 0,padtype='odd')



