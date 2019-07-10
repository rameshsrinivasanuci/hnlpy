# Mariel Tisby
# 5/21/19

# HNL HDDM Project

''' in Anaconda Prompt Type

>>> activate snakes # activates environment which contains Python 3.5
>>> cd Documents\Python_Projects
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

def main():

    # data = hddm.load_csv('/data/pdmattention/TestData.csv')
    data = hddm.load_csv('/data/pdmattention/TestData.csv')
    data.head(10) # returns first n rows for the object based on position

    # data = hddm.utils.flip_errors(data) # flips sign for lower boundary
                                        # response --> what are lower
                                        # boundaries

#    fig = plt.figure() # creates an empty figure
#    ax = fig.add_subplot(111, xlabel = 'RT', ylabel='count', title='RT distributions')

#    for i, subj_data in data.groupby('subj_idx'):

        # data.groupby splits that data into the groups based on some criterion
        # grouped = obj.groupby(key)
        # grouped = obj.groupby(key, axis = 1)
        # grouped = obj.groupby([key1, key2])

        # data.groupby('subj_idx') groups data horizontally by
        # the column subj_idx
        
        
#       for i, subj_data in data.groupby('subj_idx'):
#            subj_data.rt.hist(bins=20, histtype='step', ax=ax)

#    plt.savefig('/data/pdmattention/hddm_fig1.pdf')
#    plt.show()

    # time.sleep(5)
#    print('\n')
#    print('Done!')

    next_step(data)

def next_step(data):
#    print('working... \n')
    m = hddm.HDDM(data)
# 3 param a t v 
# v -drift
# t - nondec time -- may not depend upon diff
# a - boundary sep
# run a version with dependence of condition 
    # find a good starting point which helps with convergence
    print('finding starting values... \n')
    m.find_starting_values() # Find good starting values for optimization
                             # uses gradient ascent optimization
                             # finds the minimum of a function
    print('\n')
    print('starting values found... \n')
    # starting drawing 7000 samples and discardin 5000 as burn-in
    m.sample(5000, burn=500) # posterior samples

    print('generating stats... \n')
    stats = m.gen_stats()
    stats[stats.index.isin(['a', 'a_std', 'a_subj.0', 'a_subj.1'])]
# confirm that post for v
# confirm for all a t v for subs
    print('printing stats... \n')
    m.print_stats()

    print('\n')
    
    m.plot_posteriors(['a', 't','v'])


    m.plot_posterior_predictive(figsize=(14, 10))
    print('Legend: Red = Indiv Subject; Blue = Prediction')
    # shows how well the model fits the data
    # red is subj

    time.sleep(5)

#    models = []
    
#    for i in range(5):
#        m = hddm.HDDM(data)
#        m.find_starting_values()
#        m.sample(5000, burn=20)
#        models.append(m)

#    hddm.analyze.gelman_rubin(models)

#    m.plot_posterior_predictive(figsize=(14, 10))
if __name__ == '__main__':
    main()

# mailing list-- but when using whole data set get several error messages
# figure out

# stim -- easy, medium, hard
# 1 & 4
# 2 & 5
# 3 & 6
# separation in spatial freq
# correct -- response

# figure out about pandas structure
