import numpy
#standard make filter with b,a coefficients for teaching. 
def baselinecorrect(data,baselinesamps):
	""" 	Corrects for baseline offset in evoked potentials    
	INPUT: 	data is the data 
	OUTPUT: databc baseline corrected data 
	Automatically calculates the mean of the baseline interval
	and then removes it from the timeseries """
#
#find the mean 
	base = numpy.mean(data[baselinesamps,:],axis = 0)
	nrows = numpy.size(data,axis = 0)	
	ncols = numpy.size(data,axis = 1)	
	databc = numpy.zeros((nrows,ncols))
	for j in range(ncols):
		databc[:,j] = data[:,j] - base[j]
	return databc

