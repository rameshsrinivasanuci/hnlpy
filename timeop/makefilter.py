from scipy import signal
from matplotlib import pyplot as plt 
from numpy import log10
def makefilter(sr,fp,fs,gp=3,gs=20):
	""" 	Wrapper function around scipy filter functions.  
	Makes it convenient by providing frequency parameters in terms of 
	frequencies in Hz.   
	INPUT: 	sr - sampling rate in Hz. 
		fp - pass frequency in Hz
		fs - stop frequency in Hz
		gp - pass band ripple in dB, default 3 dB
		gs - stop band attenuation in dB, default 20 dB
		doPlot - make a plot of filter gain versus frequency, default 'no'
	OUTPUT: b,a filter coefficients.  
	Automatically detects the type of filter.  if fp < fs the filter
	is low pass but if fp > fs the filter is highpass.  """
#
#set up filter parameters

	fn = sr/2
	wp = fp/fn
	ws = fs/fn
#get the filter order

	n,wn = signal.buttord(wp,ws,gp,gs);                                                            
#design the filter

#lowpass 
	if fp < fs:
		b,a = signal.butter(n,wn,btype='lowpass')
#highpass
	if fs < fp:
		b,a = signal.butter(n,wn,btype='highpass')
#get filter respons function	
	w,h = signal.freqz(b,a,fs=sr)
	return b,a,w,h
# bode plot
