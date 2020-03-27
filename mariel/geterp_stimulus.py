from pymatreader import read_mat
import csv
import numpy
import scipy.signal as signal
import matplotlib
import matplotlib.pyplot as plt 
import timeop
from sklearn import decomposition
from scipy import linalg 
import diffusion

# globals
debug = True
lvlAnalysis = 2
path = '/data/pdmattention/task3/'
subIDs = diffusion.choose_subs(lvlAnalysis, path)
subIDs.remove('s181_ses1_')

def main():
	for subject in subIDs:
		#import data 
		currentSub = subject[0:4]
		print('Current Subject: ', currentSub)

		datadict = read_mat(path + subject + 'task3_final.mat')
		behavdict = read_mat(path + subject + '_behavior_final.mat')
		expdict = read_mat(path + subject + 'task3_expinfo.mat')

		
		#This bit was for importing ica files.  we should just calculate internally
		#ica = numpy.array(datadict['ica'])
		#organize data
		data = numpy.array(datadict['data'])

		#mix = numpy.array(datadict['mix'])
		artifact = numpy.array(datadict['artifact'])
		sr = numpy.array(datadict['sr'])
		beh_ind = numpy.array(behavdict['trial'])
		rt = numpy.array(expdict['rt'])

		# open up indices
		artifact0 = artifact.sum(axis = 0)
		artifact1 = artifact.sum(axis = 1)

		#identify goodtrials and good channels.  
		goodtrials = numpy.squeeze(numpy.array(numpy.where(artifact0 < 20)))
		goodchan = numpy.squeeze(numpy.array(numpy.where(artifact1 < 20)))


		BehEEG_int = list(set(beh_ind) & set(goodtrials))
		print('Good Trials: ', goodtrials)
		finalgoodtrials = numpy.array(BehEEG_int)
		print('finalgoodtrials', finalgoodtrials)

		#average erp. 
		erp = numpy.mean(data[:,:,finalgoodtrials],axis = 2)

		#make a lowpass filter
		sos,w,h = timeop.makefiltersos(sr,10,20)
		erpfilt = signal.sosfiltfilt(sos,erp,axis = 0,padtype='odd')

		#lets do a SVD, limiting the window in time, and taking the goodchannels. 

		u,s,vh = linalg.svd(erpfilt[1400:1625,goodchan])

		weights = numpy.zeros((129,1))

		#This is an optimal set of weights to estimate a single erp peak. 

		weights[goodchan,0] = numpy.matrix.transpose(vh[0,:])

		# Lets test it on the average time series.  

		erpfiltproject = numpy.matmul(erpfilt,weights)
		plt.figure(1, figsize=[3.5, 1.5])
		plt.clf()
		plt.plot(erpfiltproject)

		plt.xlabel('Sample')

		#now we need to apply it to every sample in the data set. 

		trialestimate = numpy.zeros((4000,360))

		for trial in finalgoodtrials:
			trialdata = numpy.squeeze(data[:,:,trial])
			trialproject = numpy.matmul(trialdata,weights)
			trialestimate[:,trial] = trialproject[:,0]

		#now we filter the single trials. 

		trialestimatefilt = signal.sosfiltfilt(sos,trialestimate,axis=0,padtype='odd')

		#lets make a plot of the average of the single trials. 

		plt.figure(2, figsize=[3.5, 1.5])
		plt.clf()
		plt.plot(numpy.mean(trialestimatefilt,axis=1))

		#lets extract two features of interest - peak value and peak timing.  

		peakvalue = numpy.zeros((360,1)) 
		peaktiming = numpy.zeros((360,1))

		for j in finalgoodtrials: 
			minval = numpy.amin(trialestimatefilt[1400:1625,j])
			index = numpy.where(trialestimatefilt[:,j] == minval)
			peaktiming[j,0] = numpy.array(index[0])
			indices = numpy.arange(index[0]-10,index[0]+10,1)
			peakvalue[j,0] = numpy.mean(trialestimatefilt[indices,j],axis=0) 
			

		#lets make scatter plots of these parameters versus rt 

		plt.figure(3, figsize=[3.5, 1.5])
		plt.clf()

		plt.scatter(peaktiming[finalgoodtrials,0],rt[finalgoodtrials],marker='o')

		plt.figure(4, figsize=[3.5, 1.5])
		plt.clf()

		plt.scatter(peakvalue[finalgoodtrials,0],rt[finalgoodtrials],marker='o')
		plt.show()

		# 1600 to 1625
		# peaktiming at 1625 then drop it 

def debugging():
	#import data 
		currentSub = 's238_ses1_'
		subject = currentSub
		print('Current Subject: ', currentSub)


		datadict = read_mat(path + subject + 'task3_final.mat')
		behavdict = read_mat(path + subject[0:4] + '_behavior_final.mat')
		expdict = read_mat(path + subject + 'task3_expinfo.mat')

		
		#This bit was for importing ica files.  we should just calculate internally
		#ica = numpy.array(datadict['ica'])
		#organize data
		data = numpy.array(datadict['data'])

		#mix = numpy.array(datadict['mix'])
		artifact = numpy.array(datadict['artifact'])
		sr = numpy.array(datadict['sr'])
		beh_ind = numpy.array(behavdict['trials'])
		rt = numpy.array(expdict['rt'])

		# open up indices
		artifact0 = artifact.sum(axis = 0)
		artifact1 = artifact.sum(axis = 1)

		#identify goodtrials and good channels.  
		goodtrials = numpy.squeeze(numpy.array(numpy.where(artifact0 < 20)))
		goodchan = numpy.squeeze(numpy.array(numpy.where(artifact1 < 20)))


		BehEEG_int = list(set(beh_ind) & set(goodtrials))
		finalgoodtrials = numpy.array(BehEEG_int)

		#average erp. 
		erp = numpy.mean(data[:,:,finalgoodtrials],axis = 2)

		#make a lowpass filter
		sos,w,h = timeop.makefiltersos(sr,10,20)
		erpfilt = signal.sosfiltfilt(sos,erp,axis = 0,padtype='odd')

		erpfiltbase = timeop.baselinecorrect(erpfilt, numpy.arange(1099,1248,1))

		#lets do a SVD, limiting the window in time, and taking the goodchannels. 

		u,s,vh = linalg.svd(erpfilt[1400:1625,goodchan])

		weights = numpy.zeros((129,1))

		#This is an optimal set of weights to estimate a single erp peak. 

		weights[goodchan,0] = numpy.matrix.transpose(vh[0,:])

		# Lets test it on the average time series.  

		erpfiltproject = numpy.matmul(erpfiltbase,weights)
		plt.figure(1, figsize=[3.5, 1.5])
		plt.clf()
		plt.plot(erpfiltproject)

		plt.xlabel('Sample')

		#now we need to apply it to every sample in the data set. 

		trialestimate = numpy.zeros((4000,360))

		for trial in finalgoodtrials:
			trialdata = numpy.squeeze(data[:,:,trial])
			trialproject = numpy.matmul(trialdata,weights)
			trialestimate[:,trial] = trialproject[:,0]

		#now we filter the single trials. 

		trialestimatefilt = signal.sosfiltfilt(sos,trialestimate,axis=0,padtype='odd')
		trialestimatefiltbase = timeop.baselinecorrect(trialestimatefiltbase, numpy.arange(1099,1248,1))

		#lets make a plot of the average of the single trials. 

		plt.figure(2, figsize=[3.5, 1.5])
		plt.clf()
		plt.plot(numpy.mean(trialestimatefilt,axis=1))

		#lets extract two features of interest - peak value and peak timing.  

		peakvalue = numpy.zeros((360,1)) 
		peaktiming = numpy.zeros((360,1))

		for j in finalgoodtrials: 
			index = numpy.where(trialestimatefilt[:,j] == minval)
			peaktiming[j,0] = numpy.array(index[0])
			indices = numpy.arange(index[0]-10,index[0]+10,1)
			peakvalue[j,0] = numpy.mean(trialestimatefilt[indices,j],axis=0) 
			

		#lets make scatter plots of these parameters versus rt 

		plt.figure(3, figsize=[3.5, 1.5])
		plt.clf()

		plt.scatter(peaktiming[finalgoodtrials,0],rt[finalgoodtrials],marker='o')

		plt.figure(4, figsize=[3.5, 1.5])
		plt.clf()

		plt.scatter(peakvalue[finalgoodtrials,0],rt[finalgoodtrials],marker='o')
		plt.show()

		return erpfiltproject, trialestimatefilt, artifact0, artifact1, artifact, data, datadict
		

if __name__ == '__main__':
	erpfiltproject, trialestimatefilt, artifact0, artifact1, artifact, data, datadict = debugging()