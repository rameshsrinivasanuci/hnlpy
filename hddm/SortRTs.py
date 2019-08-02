# Mariel Tisby 
# HNL Project: Sorting Reaction Time -- Exponetially Weighted Moving Average Filter
# 7.18.19


'''
First sort RTs from short to long
'''

'''
mar's notes:


zip_longest -- sorts all maybe??
'''

import math
import numpy as np 
import scipy.linalg as sl 



L = 1.5 #limit
l = 0.01 #lambda 
s = .5 # standard deviation 


def sortrt(rt_list):

	srtd_indices = sorted(range(len(rt_list)), key=lambda k: rt_list[k])
	rt_list.sort() 

	cutoff = 0; 

	'''
	solve numpy // matlab
	experiment with what is translating
	'''

	ucl = .5+L*s*(sl.sqrtm(l/(2-l)*(np.power((1-l), np.multiply(2, (range(len(rt_list))))))))
	lcl = .5-L*s*(sl.sqrtm(l/(2-l)*(np.power((1-l), np.multiply(2, (range(len(rt_list))))))))
	

	return rt_list, srtd_indices
