# Mariel Tisby
# HNL Project -- Checking Comparison Between Matlab and Python for RT Model
# 8/13/19

error_check = False

import numpy as np 
import h5py
from scipy.signal import lfilter
import matlab.engine as ME
import math

eng = ME.start_matlab()

L = 1.5
l = 0.01
s = .5


arrays = {}
f = h5py.File('/data/pdmattention/task3/s205_ses1_task3_expinfo.mat')
for k, v in f.items():
		arrays[k] = np.array(v)

f.close()


rt_list = []
rts =((arrays['rt']).astype(int))
cond_list = []
cond = (arrays['condition']).astype(int)
correct_list = []
correct = arrays['correct']

index = [range(0, 360)] # original indexes
index1 = []

# sort through rts and remove nan trials 
# remove nan trials for cond and correct based on rt nan trials 
for ii in range(len(rts)):

	#check_nan = math.isnan(rts[ii][0])
	#print('NUM: ', rts[ii][0], 'Index: ', ii)
	check_nan = rts[ii][0]

	if check_nan > 0: # checks for nan trials 
		rt_list.append(rts[ii][0])
		cond_list.append(cond[ii][0])
		correct_list.append(correct[ii][0])
	
	elif check_nan <= 0:
		index1.append(ii)

# EWMA method
index2 = np.argsort(rt_list) # provides sorted indicies of rts
rt_list.sort()


if error_check == True:
	print('OG: ', cond_list)
	print("OG: ", cond_list[119])

cond_list = [cond_list[i] for i in index2]

if error_check == True:
	print('New: ', cond_list)
	print('New: ', cond_list[359])

correct_list = [correct_list[i] for i in index2]


ucl = .5+L*s*(np.sqrt(l/(2-l)*(np.power((1-l), np.multiply(2, (range(len(rt_list))))))))
lcl = .5-L*s*(np.sqrt(l/(2-l)*(np.power((1-l), np.multiply(2, (range(len(rt_list))))))))

b = np.array([l])
a = np.array([1, (l-1)])
x = rts
zi = (1-l)/2

z = []
for n in range(0, len(cond)):
	
	if n == 0:
		z_add = (b[0]) * x[n][0] - (a[1]) * zi
		z_add = a[0]*z_add
		
	else:
		z_add = (b[0]) * x[n][0] - (a[1]) * z[n-1]
		z_add = a[0]*z_add
		

	z.append(round(z_add, 4)) 


z1 = np.array(z)

ucl1 = []

for ii in range(len(ucl)):
	ucl1.append(ii)

#vv = np.argwhere((np.diff(z<ucl1)== -1), 1, 'first')
eng = ME.start_matlab()
vv = eng.find(eng.diff(z<ucl1)==-1, 1, 'first')
eps = 2.2204e-16

if eng.isempty(vv) == False:
	cutoff = rt_list[vv] + eps
else:
	vv = 1

