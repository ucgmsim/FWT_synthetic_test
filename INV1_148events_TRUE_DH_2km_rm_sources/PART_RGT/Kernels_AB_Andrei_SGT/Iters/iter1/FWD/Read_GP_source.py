#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 14:19:45 2018

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy import signal
from qcore import timeseries
#import os

def readGP_2(loc, fname):
    """
    Convinience function for reading files in the Graves and Pitarka format
    """
    with open("".join([loc, fname]), 'r') as f:
        lines = f.readlines()
    
    data = []

    for line in lines[2:]:
        data.append([float(val) for val in line.split()])

    data=np.concatenate(data) 
    
    line1=lines[1].split()
    num_pts=float(line1[0])
    dt=float(line1[1])
    shift=float(line1[4])

    return data, num_pts, dt, shift

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def write_adj_source_ts(s1,v1,mainfolder,mainfolder_source,source,dt):
    #filename1=mainfolder_source+v1
    vs1=v1.split('.')
    timeseries.seis2txt(source,dt,mainfolder_source,vs1[0],vs1[1])
    return	


#_, num_pts, dt, shift  = readGP_2('','my_stf_file.txt')
_, num_pts, dt, shift  = readGP_2('','my_gaussian.txt')
t = np.arange(num_pts)*dt
############/nesi/nobackup/nesi00213/RunFolder/tdn27/rgraves/Adjoint/Syn_VMs/Kernels/#########################
fs = 1/dt
lowcut = 0.01
highcut = 0.4


fc = highcut  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
#b, a = signal.butter(4, w, 'low')
b, a = signal.butter(4, w, 'low')

#source  = timeseries.read_ascii('my_stf_file.txt')
source  = timeseries.read_ascii('my_gaussian.txt')
source_filtered = signal.filtfilt(b, a, source)
#source_est  = timeseries.read_ascii('fwd01.stf')
#
#v1='my_gaussian_filtered.txt'
#vs1=v1.split('.')
#timeseries.seis2txt(source,dt,'',vs1[0],vs1[1])

plt.figure(figsize=(10,2.5))
plt.plot(t,source,c='k')
#plt.plot(source_est,c='b')
#plt.gca().legend(('Source','Source estimated'))
plt.plot(t,source_filtered,c='b')
plt.gca().legend(('Source','Source filtered'))
plt.xlabel('Time (s)')
plt.show()


    
    
    
