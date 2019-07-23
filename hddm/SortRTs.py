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

L = 1.5 #limit
l = 0.01 #lambda 
s = .5 # standard deviation 

def sortrt(rt_list):

	srtd_indices = sorted(range(len(rt_list)), key=lambda k: rt_list[k])
	rt_list.sort() 

	cutoff = 0; 

	ucl = .5+L*s*(math.sqrt(l/(2-l)*(1 - np.power((1-l), (2*(1:len(rt_list)))))))
	return rt_list, srtd_indices
