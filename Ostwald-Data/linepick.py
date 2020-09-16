"""
# Created on 9/13/20 5:48 PM 2020 

# author: Jenny Sun 

This function shows the index and returns the index of the line when clicking
"""

import numpy as np
import matplotlib.pyplot as plt


def getline(data):
    fig = plt.figure(figsize = (11,8))
    ax = fig.add_subplot(111)
    line = plt.plot(data, picker=10)  # 5 points tolerance
    fig.canvas.mpl_connect('pick_event', onpick)

def onpick(event):
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    points = tuple(zip(xdata[ind], ydata[ind]))
    print('onpick:', thisline)


    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # plt.gcf()
    # ax.text(0.05, 0.95, thisline, transform=ax.transAxes, fontsize=14,
    #         verticalalignment='top', bbox=props)




