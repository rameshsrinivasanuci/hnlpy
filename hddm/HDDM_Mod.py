# HDDM Project
# Following the documentation from: http://ski.clps.brown.edu/hddm_docs/tutorial_python.html#loading-data
# created: 9/10/19

# Imports
import pandas as pd
import matplotlib.pyplot as plt 
import hddm

# Debugging
debug = True
#global
task = 3
model = 5;

if task ==3:
	filename1 = '/data/pdmattention/EWMAV_task3/TrainingData_task3.csv'
	filename2 = '/data/pdmattention/EWMAV_task3/TrainingData_task3_allsubs.csv'

	final_file1 = '/data/pdmattention/EWMAV_task3/task3_stats/task3_stats_'
	final_file2 = '/data/pdmattention/EWMAV_task3/task3_stats/task3_allsubs_stats_'
if task ==1:
	filename1 = '/data/pdmattention/EWMAV_task1/TrainingData_task1.csv'
	filename2 = '/data/pdmattention/EWMAV_task1/TrainingData_task1_allsubs.csv'

	final_file1 = '/data/pdmattention/EWMAV_task1/task1_stats/task1_stats_'
	final_file2 = '/data/pdmattention/EWMAV_task1/task1_stats/task1_allsubs_stats_'



if debug == True:
	print('HDDM Version: ', hddm.__version__)

data = hddm.load_csv(filename1)
print('model ', model, 'generating for task ', task)

if model ==1:
	m = hddm.HDDM(data)
	model_name = 'm1'
if model ==2:
	m = hddm.HDDM(data, depends_on={'v': 'stim'}) # drift rate depends on condition
	model_name = 'm2'
if model ==3:
	m = hddm.HDDM(data, depends_on={'a': 'stim'}) # boundary separate depends on condition
	model_name = 'm3'
if model ==4:
	m = hddm.HDDM(data, depends_on={'t': 'stim'}) # non-decision time depends on condition
	model_name = 'm4'
if model ==5:
	m = hddm.HDDM(data, depends_on={'v': 'stim', 't': 'stim'}) # both v and t depends on condition
	model_name = 'm5'

m.find_starting_values()
m.sample(2000, burn = 20)
stats = m.gen_stats()

stats.to_csv(final_file1+model_name+'.csv')