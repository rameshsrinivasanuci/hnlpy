# HDDM Project
# Following the documentation from: http://ski.clps.brown.edu/hddm_docs/tutorial_python.html#loading-data
# created: 9/10/19

# Imports
import pandas as pd
import matplotlib.pyplot as plt 
import hddm

# Debugging
debug = True

# Global Variables 
goal_dir = 'TrainingData_task3.csv'

if debug == True:
	print('HDDM Version: ', hddm.__version__)

data = hddm.load_csv(goal_dir)

m = hddm.HDDM(data)
#m.find_starting_values()
m.find_starting_values()
m.sample(2000, burn = 20)
stats = m.gen_stats()


# indexes into data frame
means = stats.loc[:, 'mean']
means.iloc[0]
