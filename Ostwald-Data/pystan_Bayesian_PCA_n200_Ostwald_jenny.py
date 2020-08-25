# pystan_Bayesian_PCA_n200.py - Testing Bayesian PCA on real ERP N200 data
#
# Copyright (C) 2020 Michael D. Nunez <mdnunez1@uci.edu>, Mariel Tisby, Ramesh Srinivasan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Record of Revisions
#
# Date            Programmers                         Descriptions of Change
# ====         ================                       ======================
# 08/18/20      Michael Nunez         Converted from pystan_Bayesian_PCA_n200.py
# 08/21/20      Michael Nunez                    Add plots of ERPs

### References ###
# https://rfarouni.github.io/assets/projects/BayesianFactorAnalysis/BayesianFactorAnalysis.html
# See stan-users-guide-2_24.pdf Section 3.4
# https://www.cs.helsinki.fi/u/sakaya/tutorial/

# Modules
import numpy as np
import pystan
import scipy.io as sio
from scipy import stats
import warnings
import os
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import linalg 
from pymatreader import read_mat
from time import strftime
### import lab modules ###
import timeop
import diffusion 


def flipstanout(insamples):
    result = {}  # Initialize dictionary
    allkeys ={} # Initialize dictionary
    keyindx = 0
    for key in insamples.keys():
        if key[0] != '_':
            possamps = insamples[key]
            transamps = np.moveaxis(possamps,0,-1)
            bettersamps = np.moveaxis(transamps,0,-1)
            if len(bettersamps.shape) == 2:
                reshapedsamps = np.reshape(bettersamps, (1,) + bettersamps.shape[0:2])
                result[key] = reshapedsamps
            else:
                result[key] = bettersamps
    return result


def diagnostic(insamples):
    """
    Returns Rhat (measure of convergence, less is better with an approximate
    1.10 cutoff) and Neff, number of effective samples).

    Reference: Gelman, A., Carlin, J., Stern, H., & Rubin D., (2004).
              Bayesian Data Analysis (Second Edition). Chapman & Hall/CRC:
              Boca Raton, FL.


    Parameters
    ----------
    insamples: dic
        Sampled values of monitored variables as a dictionary where keys
        are variable names and values are numpy arrays with shape:
        (dim_1, dim_n, iterations, chains). dim_1, ..., dim_n describe the
        shape of variable in JAGS model.

    Returns
    -------
    dict:
        Rhat, Rhatnew, Neff, posterior mean, and posterior std for each variable. Prints maximum Rhat, maximum Rhatnew, and minimum Neff across all variables
    """

    result = {}  # Initialize dictionary
    maxrhats = np.zeros((len(insamples.keys())), dtype=float)
    maxrhatsnew = np.zeros((len(insamples.keys())), dtype=float)
    minneff = np.ones((len(insamples.keys())), dtype=float)*np.inf
    allkeys ={} # Initialize dictionary
    keyindx = 0
    for key in insamples.keys():
        if key[0] != '_':
            result[key] = {}
            
            possamps = insamples[key]
            
            # Number of chains
            nchains = possamps.shape[-1]
            
            # Number of samples per chain
            nsamps = possamps.shape[-2]
            
            # Number of variables per key
            nvars = np.prod(possamps.shape[0:-2])
            
            # Reshape data
            allsamps = np.reshape(possamps, possamps.shape[:-2] + (nchains * nsamps,))

            # Reshape data to preduce R_hatnew
            possampsnew = np.empty(possamps.shape[:-2] + (int(nsamps/2), nchains * 2,))
            newc=0
            for c in range(nchains):
                possampsnew[...,newc] = np.take(np.take(possamps,np.arange(0,int(nsamps/2)),axis=-2),c,axis=-1)
                possampsnew[...,newc+1] = np.take(np.take(possamps,np.arange(int(nsamps/2),nsamps),axis=-2),c,axis=-1)
                newc += 2

            # Index of variables
            varindx = np.arange(nvars).reshape(possamps.shape[0:-2])
            
            # Reshape data
            alldata = np.reshape(possamps, (nvars, nsamps, nchains))
                    
            # Mean of each chain for rhat
            chainmeans = np.mean(possamps, axis=-2)
            # Mean of each chain for rhatnew
            chainmeansnew = np.mean(possampsnew, axis=-2)
            # Global mean of each parameter for rhat
            globalmean = np.mean(chainmeans, axis=-1)
            globalmeannew = np.mean(chainmeansnew, axis=-1)
            result[key]['mean'] = globalmean
            result[key]['std'] = np.std(allsamps, axis=-1)
            globalmeanext = np.expand_dims(
                globalmean, axis=-1)  # Expand the last dimension
            globalmeanext = np.repeat(
                globalmeanext, nchains, axis=-1)  # For differencing
            globalmeanextnew = np.expand_dims(
                globalmeannew, axis=-1)  # Expand the last dimension
            globalmeanextnew = np.repeat(
                globalmeanextnew, nchains*2, axis=-1)  # For differencing
            # Between-chain variance for rhat
            between = np.sum(np.square(chainmeans - globalmeanext),
                             axis=-1) * nsamps / (nchains - 1.)
            # Mean of the variances of each chain for rhat
            within = np.mean(np.var(possamps, axis=-2), axis=-1)
            # Total estimated variance for rhat
            totalestvar = (1. - (1. / nsamps)) * \
                within + (1. / nsamps) * between
            # Rhat (original Gelman-Rubin statistic)
            temprhat = np.sqrt(totalestvar / within)
            maxrhats[keyindx] = np.nanmax(temprhat) # Ignore NANs
            allkeys[keyindx] = key
            result[key]['rhat'] = temprhat
            # Between-chain variance for rhatnew
            betweennew = np.sum(np.square(chainmeansnew - globalmeanextnew),
                             axis=-1) * (nsamps/2) / ((nchains*2) - 1.)
            # Mean of the variances of each chain for rhatnew
            withinnew = np.mean(np.var(possampsnew, axis=-2), axis=-1)
            # Total estimated variance
            totalestvarnew = (1. - (1. / (nsamps/2))) * \
                withinnew + (1. / (nsamps/2)) * betweennew
            # Rhatnew (Gelman-Rubin statistic from Gelman et al., 2013)
            temprhatnew = np.sqrt(totalestvarnew / withinnew)
            maxrhatsnew[keyindx] = np.nanmax(temprhatnew) # Ignore NANs
            result[key]['rhatnew'] = temprhatnew
            # Number of effective samples from Gelman et al. (2013) 286-288
            neff = np.empty(possamps.shape[0:-2])
            for v in range(0, nvars):
                whereis = np.where(varindx == v)
                rho_hat = []
                rho_hat_even = 0
                rho_hat_odd = 0
                t = 2
                while ((t < nsamps - 2) & (float(rho_hat_even) + float(rho_hat_odd) >= 0)):
                    variogram_odd = np.mean(np.mean(np.power(alldata[v,(t-1):nsamps,:] - alldata[v,0:(nsamps-t+1),:],2),axis=0)) # above equation (11.7) in Gelman et al., 2013
                    rho_hat_odd = 1 - np.divide(variogram_odd, 2*totalestvar[whereis]) # Equation (11.7) in Gelman et al., 2013
                    rho_hat.append(rho_hat_odd)
                    variogram_even = np.mean(np.mean(np.power(alldata[v,t:nsamps,:] - alldata[v,0:(nsamps-t),:],2),axis=0)) # above equation (11.7) in Gelman et al., 2013
                    rho_hat_even = 1 - np.divide(variogram_even, 2*totalestvar[whereis]) # Equation (11.7) in Gelman et al., 2013
                    rho_hat.append(rho_hat_even)
                    t += 2
                rho_hat = np.asarray(rho_hat)
                neff[whereis] = np.divide(nchains*nsamps, 1 + 2*np.sum(rho_hat)) # Equation (11.8) in Gelman et al., 2013
            result[key]['neff'] = np.round(neff) 
            minneff[keyindx] = np.nanmin(np.round(neff))
            keyindx += 1

            # Geweke statistic?
    print("Maximum Rhat was %3.2f for variable %s" % (np.max(maxrhats),allkeys[np.argmax(maxrhats)]))
    print("Maximum Rhatnew was %3.2f for variable %s" % (np.max(maxrhatsnew),allkeys[np.argmax(maxrhatsnew)]))
    print("Minimum number of effective samples was %d for variable %s" % (np.min(minneff),allkeys[np.argmin(minneff)]))
    return result


def summary(insamples):
    """
    Returns parameter estimates for each posterior distribution (mean and median posteriors) as well as 95% and 99% credible intervals (.5th, 2.5th, 97.5th, 99.5th percentiles)

    Parameters
    ----------
    insamples: dic
        Sampled values of monitored variables as a dictionary where keys
        are variable names and values are numpy arrays with shape:
        (dim_1, dim_n, iterations, chains). dim_1, ..., dim_n describe the
        shape of variable in JAGS model.
    """

    result = {}  # Initialize dictionary
    maxrhats = np.zeros((len(insamples.keys())), dtype=float)
    maxrhatsnew = np.zeros((len(insamples.keys())), dtype=float)
    minneff = np.ones((len(insamples.keys())), dtype=float)*np.inf
    allkeys ={} # Initialize dictionary
    keyindx = 0
    for key in insamples.keys():
        if key[0] != '_':
            result[key] = {}
            
            possamps = insamples[key]
            
            # Number of chains
            nchains = possamps.shape[-1]
            
            # Number of samples per chain
            nsamps = possamps.shape[-2]
            
            # Number of variables per key
            nvars = np.prod(possamps.shape[0:-2])
            
            # Reshape data
            allsamps = np.reshape(possamps, possamps.shape[:-2] + (nchains * nsamps,))

            # Reshape data to preduce R_hatnew
            possampsnew = np.empty(possamps.shape[:-2] + (int(nsamps/2), nchains * 2,))
            newc=0
            for c in range(nchains):
                possampsnew[...,newc] = np.take(np.take(possamps,np.arange(0,int(nsamps/2)),axis=-2),c,axis=-1)
                possampsnew[...,newc+1] = np.take(np.take(possamps,np.arange(int(nsamps/2),nsamps),axis=-2),c,axis=-1)
                newc += 2

            result[key]['mean'] = np.mean(allsamps, axis=-1)
            result[key]['std'] = np.std(allsamps, axis=-1)
            result[key]['median'] = np.quantile(allsamps,0.5, axis=-1)
            result[key]['95lower'] = np.quantile(allsamps,0.025, axis=-1)
            result[key]['95upper'] = np.quantile(allsamps,0.975, axis=-1)
            result[key]['99lower'] = np.quantile(allsamps,0.005, axis=-1)
            result[key]['99upper'] = np.quantile(allsamps,0.995, axis=-1)
    return result


# def jellyfish(possamps):  # jellyfish plots
#     """Plots posterior distributions of given posterior samples in a jellyfish
#     plot. Jellyfish plots are posterior distributions (mirrored over their
#     horizontal axes) with 99% and 95% credible intervals (currently plotted
#     from the .5% and 99.5% & 2.5% and 97.5% percentiles respectively.
#     Also plotted are the median and mean of the posterior distributions"

#     Parameters
#     ----------
#     possamps : ndarray of posterior chains where the last dimension is
#     the number of chains, the second to last dimension is the number of samples
#     in each chain, all other dimensions describe the shape of the parameter
#     """

#     # Number of chains
#     nchains = possamps.shape[-1]

#     # Number of samples per chain
#     nsamps = possamps.shape[-2]

#     # Number of dimensions
#     ndims = possamps.ndim - 2

#     # Number of variables to plot
#     nvars = np.prod(possamps.shape[0:-2])

#     # Index of variables
#     varindx = np.arange(nvars).reshape(possamps.shape[0:-2])

#     # Reshape data
#     alldata = np.reshape(possamps, (nvars, nchains, nsamps))
#     alldata = np.reshape(alldata, (nvars, nchains * nsamps))

#     # Plot properties
#     LineWidths = np.array([2, 5])
#     teal = np.array([0, .7, .7])
#     blue = np.array([0, 0, 1])
#     orange = np.array([1, .3, 0])
#     Colors = [teal, blue]

#     # Initialize ylabels list
#     ylabels = ['']

#     for v in range(0, nvars):
#         # Create ylabel
#         whereis = np.where(varindx == v)
#         newlabel = ''
#         for l in range(0, ndims):
#             newlabel = newlabel + ('_%i' % whereis[l][0])

#         ylabels.append(newlabel)

#         # Compute posterior density curves
#         kde = stats.gaussian_kde(alldata[v, :])
#         bounds = stats.scoreatpercentile(alldata[v, :], (.5, 2.5, 97.5, 99.5))
#         for b in range(0, 2):
#             # Bound by .5th percentile and 99.5th percentile
#             x = np.linspace(bounds[b], bounds[-1 - b], 100)
#             p = kde(x)

#             # Scale distributions down
#             maxp = np.max(p)

#             # Plot jellyfish
#             upper = .25 * p / maxp + v + 1
#             lower = -.25 * p / maxp + v + 1
#             lines = plt.plot(x, upper, x, lower)
#             plt.setp(lines, color=Colors[b], linewidth=LineWidths[b])
#             if b == 1:
#                 # Mark mode
#                 wheremaxp = np.argmax(p)
#                 mmode = plt.plot(np.array([1., 1.]) * x[wheremaxp],
#                                  np.array([lower[wheremaxp], upper[wheremaxp]]))
#                 plt.setp(mmode, linewidth=3, color=orange)
#                 # Mark median
#                 mmedian = plt.plot(np.median(alldata[v, :]), v + 1, 'ko')
#                 plt.setp(mmedian, markersize=10, color=[0., 0., 0.])
#                 # Mark mean
#                 mmean = plt.plot(np.mean(alldata[v, :]), v + 1, '*')
#                 plt.setp(mmean, markersize=10, color=teal)

#     # Display plot
#     plt.setp(plt.gca(), yticklabels=ylabels, yticks=np.arange(0, nvars + 1))


# def recovery(possamps, truevals):  # Parameter recovery plots
#     """Plots true parameters versus 99% and 95% credible intervals of recovered
#     parameters. Also plotted are the median and mean of the posterior
#     distributions

#     Parameters
#     ----------
#     possamps : ndarray of posterior chains where the last dimension is the
#     number of chains, the second to last dimension is the number of samples in
#     each chain, all other dimensions must match the dimensions of truevals

#     truevals : ndarray of true parameter values
#     """

#     # Number of chains
#     nchains = possamps.shape[-1]

#     # Number of samples per chain
#     nsamps = possamps.shape[-2]

#     # Number of variables to plot
#     nvars = np.prod(possamps.shape[0:-2])

#     # Reshape data
#     alldata = np.reshape(possamps, (nvars, nchains, nsamps))
#     alldata = np.reshape(alldata, (nvars, nchains * nsamps))
#     truevals = np.reshape(truevals, (nvars))

#     # Plot properties
#     LineWidths = np.array([2, 5])
#     teal = np.array([0, .7, .7])
#     blue = np.array([0, 0, 1])
#     orange = np.array([1, .3, 0])
#     Colors = [teal, blue]

#     for v in range(0, nvars):
#         # Compute percentiles
#         bounds = stats.scoreatpercentile(alldata[v, :], (.5, 2.5, 97.5, 99.5))
#         for b in range(0, 2):
#             # Plot credible intervals
#             credint = np.ones(100) * truevals[v]
#             y = np.linspace(bounds[b], bounds[-1 - b], 100)
#             lines = plt.plot(credint, y)
#             plt.setp(lines, color=Colors[b], linewidth=LineWidths[b])
#             if b == 1:
#                 # Mark median
#                 mmedian = plt.plot(truevals[v], np.median(alldata[v, :]), 'o')
#                 plt.setp(mmedian, markersize=10, color=[0., 0., 0.])
#                 # Mark mean
#                 mmean = plt.plot(truevals[v], np.mean(alldata[v, :]), '*')
#                 plt.setp(mmean, markersize=10, color=teal)
#     # Plot line y = x
#     tempx = np.linspace(np.min(truevals), np.max(
#         truevals), num=100)
#     recoverline = plt.plot(tempx, tempx)
#     plt.setp(recoverline, linewidth=3, color=orange)

def mm2inch(*tupl):
    mmperinch = 25.4
    if isinstance(tupl[0], tuple):
        return tuple(i/mmperinch for i in tupl[0])
    else:
        return tuple(i/mmperinch for i in tupl)


# get the first 4 runs for sub-006
import preprocess_ostwald as po
subID = 'sub-006'
run1 = po.get_epoch(subID,'01')
run2 = po.get_epoch(subID,'02')
run3 = po.get_epoch(subID,'03')
run4 = po.get_epoch(subID,'04')
run5 = po.get_epoch(subID,'05')

grand = []

grand = np.append(run1['trialeeg'],run2['trialeeg'],axis = 2)
grand = np.append(grand, run3['trialeeg'],axis = 2)
grand = np.append(grand, run4['trialeeg'],axis = 2)
grand = np.append(grand, run5['trialeeg'],axis = 2)

condition = [] 

condition = list(np.array(run1['condition'])[:,1]) + list(np.array(run2['condition'])[:,1]) + \
        list(np.array(run3['condition'])[:,1]) + list(np.array(run4['condition'])[:,1]) + \
        list(np.array(run5['condition'])[:,1])

# construct a channel x trial matrix
channelnum = grand.shape[1]
trialnum = grand.shape[2]
artifact = np.zeros((channelnum, trialnum))

# for each channel in each trial, mark 1 if the abs(mV) is more than 100
for trial in range(0, trialnum):
    for chan in range(0, channelnum):
        waveform = grand[:, chan, trial]
        if all(abs(i) <= 100 for i in waveform) is True:
            artifact[chan, trial] = 0
        else:
            artifact[chan, trial] = 1

goodtrials, goodchan = po.get_indices(artifact)

data=[]

condition = np.array(condition)




#Bayesian PCA (see https://www.cs.helsinki.fi/u/sakaya/tutorial/)
#This would reuce to regular PCA if Psi was isotropic, that is, proportional to the identity matrix (all the diagonal elements are equal)


### Global Variables ###
'''
debug | for debugging code. Use if debug == True
lvlAnalysis | lvlAnalysis indicates if we are running analysis on test data or all data
            1 = test
            2 = all
path | path where data is
'''
debug = True
lvlAnalysis = 2
path = '/home/ramesh/pdmattention/task3/'
subID = 'sub-016'

currentSub = subID[0:4]
print('Current Subject: ', currentSub)
datadict = read_mat('/home/jenny/ostwald-data/clean-eeg-converted/sub-016_run-03_info.mat')
behavdict = read_mat(path + subID[0:4] + '_behavior_final.mat')
expdict = read_mat(path + subID + 'task3_expinfo.mat')
d

data = np.array(datadict['data'])

artifact = np.array(datadict['artifact'])
sr = np.array(datadict['sr'])
beh_ind = np.array(behavdict['trials'])

# open up indices
artifact0 = artifact.sum(axis = 0)
artifact1 = artifact.sum(axis = 1)

identify good trials and good channels.
goodtrials = np.squeeze(np.array(np.where(artifact0 < 20)))
goodchan = np.squeeze(np.array(np.where(artifact1 < 40)))

goodtrials = np.array(datadict['goodtrials'])
goodchan = np.array(datadict['goodchan'])


# BehEEG_int = list(set(beh_ind) & set(goodtrials))
finalgoodtrials = np.array(diffusion.compLists(beh_ind, goodtrials))
# finalgoodtrials = np.array(BehEEG_int)

## separate by condition ###
# separate finalgoodtrials by condition



condition = np.take(condition, goodtrials)

finalgoodtrials1 = []
finalgoodtrials2 = []
finalgoodtrials3 = []
finalgoodtrials4 = []

count = 0
for value in condition:
    if value == 1:
        finalgoodtrials1.append(goodtrials[count])
    elif value == 2:
        finalgoodtrials2.append(goodtrials[count])
    elif value == 3:
        finalgoodtrials3.append(goodtrials[count])
    elif value == 4:
        finalgoodtrials4.append(goodtrials[count])
    count = count + 1
            
finalgoodtrialList = [finalgoodtrials1, finalgoodtrials2, finalgoodtrials3, finalgoodtrials4]        

data = grand[:,:,:]

allYinput = []
fullwindowERPs = []
for triallist in finalgoodtrialList:
    finalgoodtrial = triallist
    		#average erp.
    erp = np.mean(data[:, :, goodtrials], axis=2)
    # plt.plot(np.arange(-200, 1000, 2), erp)
    
        #make a lowpass filter
    sr = 500
    sos,w,h = timeop.makefiltersos(sr,10,20)
    erpfilt = signal.sosfiltfilt(sos,erp,axis = 0,padtype='odd')
    
    erpfiltbase = timeop.baselinecorrect(erpfilt, np.arange(924,998,1))
    #lets do a SVD, limiting the window in time, and taking the goodchannels. 
    
    # plt.plot(erpfiltbase)
    # plt.show()

    Y = erpfiltbase[1075:1288, goodchan]
    # Y = erpfiltbase[1075:1288,goodchan]
    allYinput.append(Y)
    fullwindowERPs.append(erpfiltbase[:,goodchan])
    
Y1 = ( allYinput[0] - np.mean(allYinput[0].flatten()) ) / np.std(allYinput[0].flatten()) #Standardize
Y2 = ( allYinput[1] - np.mean(allYinput[1].flatten()) ) / np.std(allYinput[1].flatten()) #Standardize
Y3 = ( allYinput[2] - np.mean(allYinput[2].flatten()) ) / np.std(allYinput[2].flatten()) #Standardize
Y4 = ( allYinput[3] - np.mean(allYinput[3].flatten()) ) / np.std(allYinput[2].flatten()) #Standardize

N = Y1.shape[0] # Number of time points
D = 4 #Number of components
P = Y1.shape[1] # Number of good channels
M  = int(D*(P-D)+ D*(D-1)/2)  # number of non-zero loadings (Lower triangle of L), lower rectangle area + upper lower triangle

# Stan code
Bayesian_PCA_code = """
data {
  int<lower=1> N;                // number of time points
  int<lower=1> P;                // number of components
  matrix[N,P] Y1;                 // Condition 1 EEG matrix of order [N,P]
  matrix[N,P] Y2;                 // Condition 2 EEG matrix of order [N,P]
  matrix[N,P] Y3;                 // Condition 3 EEG matrix of order [N,P]
  matrix[N,P] Y4;                 // Condition 4 EEG matrix of order [N,P]
  int<lower=1> D;              // number of good channels
}
transformed data {
  int<lower=1> M;
  vector[P] mu;
  M  = D*(P-D)+ D*(D-1)/2;  // number of non-zero loadings (Lower triangle of L), lower rectangle area + upper lower triangle
  mu = rep_vector(0.0,P);
}
parameters {    
  vector[M] L_t;   // lower triangular elements of L (latent factors)
  vector<lower=0>[D] L_d;   // diagonal elements of L
  vector<lower=0>[P] psi;         // vector of error variances, truncated at 0 for half-cauchy prior
  real<lower=0>   mu_psi; // hierarchical mean of error variances, truncated at 0 for half-cauchy prior
  real<lower=0>  sigma_psi; // hierarchical std of error variances, truncated at 0 for half-cauchy prior
  real   mu_lt; // hierarchical mean of lower triangular elements of L (latent factors)
  real<lower=0>  sigma_lt; // hierarchical std of lower triangular elements of L (latent factors)
}
transformed parameters{
  cholesky_factor_cov[P,D] L;  //lower triangular factor loadings Matrix 
  cov_matrix[P] Q;   //Covariance matrix of Y
{
  int idx1;
  int idx2;
  real zero;
  idx1 = 0; 
  idx2 = 0;
  zero = 0;
  for(i in 1:P){
    for(j in (i+1):D){
      idx1 = idx1 + 1;
      L[i,j] = zero; //constrain the upper triangular elements to zero 
    }
  }
  for (j in 1:D) {
      L[j,j] = L_d[j]; //Place diagonal elements of latent factor matrix in the matrix
    for (i in (j+1):P) {
      idx2 = idx2 + 1;
      L[i,j] = L_t[idx2]; //Place the lower triangle elements of latent factor matrix in the matrix
    } 
  }
} 
Q=L*L'+diag_matrix(psi); 
}
model {
// the hyperpriors 
   mu_psi ~ cauchy(0, 1);
   sigma_psi ~ cauchy(0,1);
   mu_lt ~ cauchy(0, 1);
   sigma_lt ~ cauchy(0,1);
// the priors 
  L_d ~ cauchy(0,3);
  L_t ~ cauchy(mu_lt,sigma_lt);
  psi ~ cauchy(mu_psi,sigma_psi);
//The likelihood
  for( j in 1:N){
      Y1[j] ~ multi_normal(mu,Q); 
      Y2[j] ~ multi_normal(mu,Q); 
      Y3[j] ~ multi_normal(mu,Q); 
      Y4[j] ~ multi_normal(mu,Q); 
  }
}
"""

# pystan code with Variational Bayes

latent_factor_data = {'P': P, 'N': N, 'Y1': Y1, 'Y2': Y2, 'Y3': Y3, 'Y4': Y4,'D': D}

savedir = '/home/jenny/hnlpyjenny/Ostwald-Data/model_fits/'
modelname = 'Bayesian_PCA_n200'

# Save model
timestart = strftime('%b') + '_' + strftime('%d') + '_' + \
    strftime('%y') + '_' + strftime('%H') + '_' + strftime('%M')
filename = modelname + timestart
modelfile = filename + '.stan'
f = open(savedir+modelfile, 'w')
f.write(Bayesian_PCA_code)
f.close()
print('Fitting model %s ...' % (modelfile))

sm = pystan.StanModel(model_code=Bayesian_PCA_code)

vbfit = sm.vb(data=latent_factor_data,pars=['L','psi','sigma_psi','mu_psi','sigma_lt','mu_lt'],seed=2020)

niters = np.array(vbfit['sampler_params'][0]).shape[0]

samples = dict()
samples['L'] = np.zeros((P,D,niters,1))
Lindex = 0
for d in range(D):
    for p in range(P):
        Lindex = p + d*P
        samples['L'][p,d,:,0] = np.array(vbfit['sampler_params'][Lindex])

samples['comp1_weights'] = np.mean(samples['L'][:,0,:,0].squeeze(),axis=1)
samples['comp2_weights'] = np.mean(samples['L'][:,1,:,0].squeeze(),axis=1)
samples['comp3_weights'] = np.mean(samples['L'][:,2,:,0].squeeze(),axis=1)
samples['comp4_weights'] = np.mean(samples['L'][:,3,:,0].squeeze(),axis=1)

savestring = filename + ".mat"
print('Saving results to: \n %s' % (savestring))
sio.savemat(savedir + savestring, samples)

print(vbfit)


## Plot Bayesian PCA components in the time domain
# samples = sio.loadmat('model_fits/Bayesian_PCA_n200Aug_18_20_14_38.mat');
samples = sio.loadmat(savedir + savestring);
#Plot only the first condition ERPs
fullwindowComp1 = np.matmul(fullwindowERPs[0],samples['comp1_weights'].T)
fullwindowComp2 = np.matmul(fullwindowERPs[0],samples['comp2_weights'].T)
fullwindowComp3 = np.matmul(fullwindowERPs[0],samples['comp3_weights'].T)
fullwindowComp4 = np.matmul(fullwindowERPs[0],samples['comp4_weights'].T)
window = np.arange(-300,700,2) # Time point 0 is the onset of the visual stimulus

fig, axs = plt.subplots(4, 1,figsize=mm2inch(244,110), sharex='col')
y= np.zeros(len(window))
axs[0].plot(window,fullwindowERPs[0][850:1350])
line1 = axs[0].plot(np.arange(-300,700,2),y,'k')

ax = line1[0].axes
plt.xticks(list(plt.xticks()[0]))
major_ticks = LineTicks(traj, range(0, n, 10), 10, lw=2,
label=['{:.2f} s'.format(tt) for tt in t[::10]])


axs[0].tick_params(axis="x", direction="in")
axs[0].axvline(0,color='black') # y = 0
# axs[0].get_yaxis().set_visible(False)
axs[1].plot(window,fullwindowComp1[850:1350])
# axs[1].get_yaxis().set_visible(False)
axs[2].plot(window,-fullwindowComp2[850:1350])
# axs[2].get_yaxis().set_visible(False)
axs[3].plot(window,-fullwindowComp3[850:1350])
# axs[3].get_yaxis().set_visible(False)
axs[3].set_xlabel('Time after stimulus onset (ms)', fontsize=14)
axs[0].set_title('Original Component')

fig.set_size_inches(mm2inch(218,147),forward=False)
# plt.savefig((f'model_fits/{subID}Bayesian_PCA_N200_timecourse.png'), dpi=300, format='png',bbox_inches='tight')
plt.savefig(('/home/jenny/hnlpyjenny/Ostwald-Data/{subID}Bayesian_PCA_N200_time.png'), dpi=300, format='png',bbox_inches='tight')


# la = fit.extract(permuted=False, pars=['L','psi','sigma_psi','mu_psi','sigma_lt','mu_lt'])  # return a dictionary of arrays that can be used by functions above

# samples = flipstanout(la)

# savestring = filename + ".mat"
# print('Saving results to: \n %s' % (savestring))
# sio.savemat(savedir + savestring, samples)

# diags = diagnostic(samples)
# sumstats = summary(samples)

# plt.figure()
# recovery(samples['L'],L)

# plt.figure()
# recovery(samples['psi'],np.diag(Psi))

# plt.figure()
# jellyfish(samples['sigma_psi'])

# plt.figure()
# jellyfish(samples['mu_psi'])

# plt.figure()
# jellyfish(samples['sigma_lt'])

# plt.figure()
# jellyfish(samples['mu_lt'])



# weights = np.zeros((129,1))

# #This is an optimal set of weights to estimate a single erp peak. 

# weights[goodchan,0] = np.matrix.transpose(vh[0,:])

# 		# Lets test it on the average time series.  
# erpfiltproject = np.matmul(erpfiltbase,weights)



#     #Let's find the peak timing and amplitude in the subject average.  
# erpmin = np.amin(erpfiltproject[1400:1625])
# erpmax = np.amax(erpfiltproject[1400:1625])        
# if abs(erpmin) < abs(erpmax):
#     weights = -weights
#     erpfiltproject = -erpfiltproject
    
# erp_peaktiming = np.argmin(erpfiltproject[1400:1625])+1400
# indices = np.arange(erp_peaktiming-10,erp_peaktiming+10,1)
# erp_peakvalue = np.mean(erpfiltproject[indices])
# 		#now we need to apply it to every sample in the data set. 

# trialestimate = np.zeros((4000,360))

# for trial in finalgoodtrials:
#     trialdata = np.squeeze(data[:,:,trial])
#     trialproject = np.matmul(trialdata,weights)
#     trialestimate[:,trial] = trialproject[:,0]

# 		#now we filter the single trials. 

# trialestimatefilt = signal.sosfiltfilt(sos,trialestimate,axis=0,padtype='odd')
# trialestimatefiltbase = timeop.baselinecorrect(trialestimatefilt, np.arange(1099,1248,1))

# peakvalue = np.zeros(360) 
# peaktiming = np.zeros(360)

# for j in finalgoodtrials: 
#     peaktiming[j] = np.argmin(trialestimatefiltbase[1400:1625,j])+1400
#     indices = np.arange(peaktiming[j]-10,peaktiming[j]+10,1)
#     peakvalue[j] = np.mean(trialestimatefiltbase[indices.astype(int),j],axis=0) 


# 		#lets make scatter plots of these parameters versus rt 
# trialestimate = trialestimatefiltbase

# n200 = dict();
# n200['peakvalue'] = peakvalue
# n200['peaktiming'] = peaktiming
# n200['erp_peakvalue'] = erp_peakvalue
# n200['erp_peaktiming'] = erp_peaktiming
# n200['singletrial'] = trialestimate
# n200['weights'] = weights
# n200['erp_project'] = erpfiltproject
# n200['erp'] = erpfiltbase
# n200['goodtrials'] = finalgoodtrials
# n200['goodchannels'] = goodchan

# allN200.append(n200)

###     
# subIDs = diffusion.choose_subs(lvlAnalysis, path)
# subIDs.remove('s181_ses1_')
# subIDs.remove('s193_ses2_')
# subIDs2 = ['s181_ses1_']

# for subID in subIDs2:
#     path1 = '/home/mariel/Documents/Projects2/testingParam/testingN200/'
#     n200ALL = n200AmpLatbyCond(subID)
#     n200 = dict()
#     n200['condition1'] = n200ALL[0]
#     n200['condition2'] = n200ALL[1]
#     n200['condition3'] = n200ALL[2]
#     outname = path+subID+'N200_BayesianPCA.mat'
#     savemat(outname,n200)
         
# for subID in subIDs:
#     n200ALL = n200AmpLatbyCond(subID)
#     n200 = dict()
#     n200['condition1'] = n200ALL[0]
#     n200['condition2'] = n200ALL[1]
#     n200['condition3'] = n200ALL[2]
#     outname = path+subID+'N200_BayesianPCA.mat'
#     savemat(outname,n200)
