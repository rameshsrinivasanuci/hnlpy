# Mariel Tisby
# HNL Project -- Checking Comparison Between Matlab and Python for RT Model
# 8/13/19

error_check = False

import numpy as np 
import h5py
from scipy.signal import lfilter
import matlab.engine as ME
import math
import matplotlib.pyplot as plt
import os

eng = ME.start_matlab()

L = 1.5
l = 0.01
s = .5

arrays = {}
f = h5py.File( '/data/pdmattention/task3/s205_ses1_task3_expinfo.mat')
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
		rt_list.append(rts[ii][0].item()/1000)
		cond_list.append(cond[ii][0].item())
		correct_list.append(correct[ii][0].item())
	
	elif check_nan <= 0:
		index1.append(ii)



# EWMA method
index2 = np.argsort(rt_list) # provides sorted indicies of rts
rt_list.sort()


if error_check == True:
	print('OG: ', cond_list)
	print("OG: ", cond_list[1])

cond_list = [cond_list[i] for i in index2]

if error_check == True:
	print('New: ', cond_list)
	print('New: ', cond_list[3])

correct_list = [correct_list[i] for i in index2]

check1 = np.power((1-l), np.multiply(2,(range(len(rt_list)))))
check1 = np.round(check1,4)
check4 = 1-check1
check2 = np.round((l/(2-l)), 4)
check3 = np.round(check4*check2,4)
check5 = np.round(np.sqrt(check3), 4)

ucl = .5+L*s*check5
lcl = .5-L*s*check5
#ucl = .5+L*s*(np.sqrt(l/(2-l)*(np.power((1-l), np.multiply(2, (range(len(cond_list))))))))
#lcl = .5-L*s*(np.sqrt(l/(2-l)*(np.power((1-l), np.multiply(2, (range(len(cond_list))))))))

b = np.array([l])
a = np.array([1, (l-1)])
x = np.array(correct_list)
zi = np.array((1-l)/2)

z = []
for n in range(len(x)):
	
	if n == 0:
		z_add = (b[0]) * x[n]- (a[1]) * zi
		z_add = a[0]*z_add
		z.append(round(z_add, 4)) 
	else:
		z_add = (b[0]) * x[n] - (a[1]) * z[n-1]
		z_add = a[0]*z_add
		z.append(round(z_add, 4)) 


z1 = np.array(z)

ucl1 = []

for ii in range(len(ucl)):
	ucl1.append(ii)

#vv = np.argwhere((np.diff(z<ucl1)== -1), 1, 'first')
eng = ME.start_matlab()

# vvv have to break into two lines of code because np.diff returns bool instead of 1s and 0s vvv
# vv = eng.find(np.diff(z1<ucl)==-1, 1, 'first')

Diff = ((np.diff(z1<ucl) == -1).astype(int))

if np.count_nonzero(Diff)==0:
	vv = []
else:
	vv1 = np.where(Diff>0)
	vv = vv1[0].item()

eps = 2.2204e-16


cutoff = 0

if eng.isempty(vv) == False:
	cutoff = rt_list[vv] + eps
else:
	vv = 1
'''
# figure out if var for YBlue and XBlue can be passed as python int list 
# then use eng.plot(ii) to plot everything 

BlueX = [cutoff]
for ii in range(len(rt_list)):
	if rt_list[ii] > cutoff:
		BlueX.append(rt_list[ii])


BlueY = [((ucl[vv]).astype(int)).item()]
for ii in range(len(rt_list)):
	if rt_list[ii] > cutoff:
		BlueY.append(z[ii])


RedX = []
for ii in range(len(rt_list)):
	if rt_list[ii] < cutoff:
		RedX.append(rt_list[ii])

RedX.append(cutoff)

RedY = []
for ii in range(len(rt_list)):
	if rt_list[ii] < cutoff:
		RedY.append(z[ii])
	
RedY.append(((ucl[vv]).astype(int)).item())


BlueX = [cutoff]
RedX = []
for ii in range(len(rt_list)):
	if rt_list[ii] > cutoff:
		BlueX.append(rt_list[ii])
	elif rt_list[ii]<cutoff:
		RedX.append(rt_list[ii])


BlueY = [((ucl[vv]).astype(int)).item()]
RedY = []
for ii in range(len(rt_list)):
	if rt_list[ii] > cutoff:
		BlueY.append(z[ii])
	elif rt_list[ii]<cutoff:
		RedY.append(z[ii])

check = len(BlueX) + len(RedX)

if check < 500:
	bf = 'b.'
	rf = 'r.'
else:
	bf = 'b-'
	rf = 'r-'
'''
rt_1 = []
rt_2 = []
z_1 = []
z_2 = []

slope1_list = []
slope2_list = []
for ii in range(len(rt_list)):
	if z1[ii] > ucl[ii]:
		rt_1.append(rt_list[ii])
		z_1.append(z[ii])
	else:
		rt_2.append(rt_list[ii])
		z_2.append(z[ii])

print('percentile: ', np.percentile(rt_list, 5))
percentile = np.percentile(rt_list,5).item()

rt_list2 = []
for ii in rt_list:
	rt_list2.append(round(ii, 2))


xx = rt_list2.index(round(percentile, 2))

plt.plot(rt_1, z_1, 'b.')
plt.plot(rt_2, z_2, 'r.')
plt.plot(rt_list, ucl, 'm:')
plt.plot(rt_list2[xx], z[xx], 'yv')
'''
plt.plot(BlueX, BlueY, bf)
plt.plot(RedX, RedY, rf)
#plt.fill([rt_list, rt_list.reverse()], [ucl, lcl], 'r' )
'''

plt.xlim(0.3, 1.4)
plt.show(block=True)
