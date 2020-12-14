#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 16:05:45 2020

@author: jenny
"""
# this script put together some features to prepare the data for fcn

rootdir = '/home/ramesh/pdmattention/ssvep/test'
data30 = np.load(rootdir + '/data.npy')
data40 = np.load(rootdir + '/data_40.npy')
data_twofreq = np.hstack((data30,data40))
np.save(rootdir + '/data_twofreq', data_twofreq)

target30 = np.load(rootdir + '/target.npy')
target40 = np.load(rootdir + '/target_40.npy')
target_twofreq = np.hstack((target30,target40))
np.save(rootdir + '/target_twofreq', target_twofreq)

targetrt_30 = np.load(rootdir + '/target_rt.npy')
targetrt_40 = np.load(rootdir + '/target_rt_40.npy')
targetrt_twofreq = np.hstack((targetrt_30,targetrt_40))
np.save(rootdir + '/targetrt_twofreq', targetrt_twofreq)
