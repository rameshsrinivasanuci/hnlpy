# Mariel Tisby
# 8/15/19


from scipy.signal import lfilter
import matlab.engine as ME
import math

eng = ME.start_matlab()

L = 1.5
l = 0.01
s = .5

def EWMAV(data):
	rts = data[0]

	all_rt = []
	for xx in range(len(rts)):
		x = rts[xx]
		for ii in range(len(x)):
			all_rt.append(rts[xx][ii])


	for ii in range(len(all_rt)):

	#check_nan = math.isnan(rts[ii][0])
	#print('NUM: ', rts[ii][0], 'Index: ', ii)
	check_nan = all_rt[ii][0]

	if check_nan > 0: # checks for nan trials 
		rt_list.append(all_rt[ii][0])
		cond_list.append(cond[ii][0])
		correct_list.append(correct[ii][0])
	
	elif check_nan <= 0:
		index1.append(ii)

	return all_rt