# Mariel Tisby
# 5/21/19

# HNL HDDM Project

''' in Anaconda Prompt Type

>>> activate snakes # activates environment which contains Python 3.5
>>> cd Documents Python_Projects
>>> python HDDMPrac.py

'''
import time # for readers use to pause program
import pandas as pd
import matplotlib.pyplot as plt # matplotlib is a 2D plotting library
                                # matplotlib.pyplot is a collection of
                                # comannd style functions that make
                                # matplotlib work like MATLAB
# aliasing modules
# import [module] as [another name]

print('working...')
print(' ')

import hddm
print('HDDM VER:', hddm.__version__) # check

data = hddm.load_csv('/data/pdmattention/TrainingData_task3.csv')
data.head(10) # returns first n rows for the object based on position

m = hddm.HDDM(data)
m.save('/data/pdmattention/EWMAV_task3')
    #m = hddm.HDDM(data, depends_on={'t','condition'})
    #m = hddm.HDDM(data, depends_on={'a','correct'})
    #m = hddm.HDDM(data, depends_on={'v','rt'})
print('finding starting values... \n')
m.find_starting_values() # Find good starting values for optimization
                             # uses gradient ascent optimization
                             # finds the minimum of a function
print('\n')
print('starting values found... \n')
# starting drawing 7000 samples and discardin 5000 as burn-in
m.sample(5000, burn=500) # posterior samples
#m.sample(1000, burn=20) # smaller sample to test
print('generating stats... \n')
stats = m.gen_stats()

print('stats type: ', type(stats))

stats[stats.index.isin(['a', 'a_std', 'a_subj.0', 't', 't_std', 't_subj.0', 'v', 'v_std', 'v_subj.0'])]

print('printing stats... \n')
m.print_stats()


