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


#Error Checking Filter Function ~ Not Working
x1 = np.array([l])
x2 = np.array([1, (1-l)])
x3 = np.array([(1-l)/2])

print(type(x1), 'x1', type(x2), 'x2', type(x3), 'x3')

#z = lfilter(np.array([l]), np.array([1, (1-l)]), rts, zi= np.array([(1-l)/2]))

# variables defined for filtering

b = np.array([l])
a = np.array([1, (l-1)])
x = rts
zi = (1-l)/2

z = []
for n in range(0, len(cond)):
	print('n: ', n)
	if n == 0:
		z_add = (b[0]) * x[n][0] - (a[1]) * zi
		z_add = a[0]*z_add
		
	else:
		z_add = (b[0]) * x[n][0] - (a[1]) * z[n-1]
		z_add = a[0]*z_add
		

	z.append(round(z_add, 4)) 
