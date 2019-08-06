# Mariel Tisby
# HNL Project -- Checking Comparison Between Matlab and Python for RT Model
# 8/2/19
import numpy as np 
import h5py
from scipy.signal import lfilter

L = 1.5
l = 0.01
s = .5


arrays = {}
f = h5py.File('/data/pdmattention/task3/s205_ses1_task3_expinfo.mat')
for k, v in f.items():
		arrays[k] = np.array(v)

f.close()

rt_list = []
rts = arrays['rt']
cond_list = []
cond = (arrays['condition']).astype(int)

for ii in range(len(rts)):
	rt_list.append(rts[ii][0])
	cond_list.append(cond[ii][0])


check1 = np.power((1-l), np.multiply(2,(range(len(cond_list)))))
check4 = 1-check1
check2 = l/((2-l))
check3 = check4*check2
check5 = np.round(np.sqrt(check3), 4)


######
X = cond_list

ucl = .5+L*s*(np.sqrt(l/(2-l)*(np.power((1-l), np.multiply(2, (range(len(cond_list))))))))
lcl = .5-L*s*(np.sqrt(l/(2-l)*(np.power((1-l), np.multiply(2, (range(len(cond_list))))))))

z = lfilter(l, [1, (1-l)], cond, zi=((1-l)/2))
