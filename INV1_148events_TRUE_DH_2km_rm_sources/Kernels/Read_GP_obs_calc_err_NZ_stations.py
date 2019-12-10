#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 14:19:45 2018

@author: user
"""

import numpy as np
#import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy import signal
from scipy import integrate
from qcore import timeseries

def read_source(source_file):
    
    with open(source_file, 'r') as f:
        lines = f.readlines()    
    line0=lines[0].split()
    nShot=int(line0[0])
    S=np.zeros((nShot,3))
    for i in range(1,nShot+1):
        line_i=lines[i].split()
        S[i-1,0]=line_i[0]
        S[i-1,1]=line_i[1]
        S[i-1,2]=line_i[2]
    
    return nShot, S

def readGP_2(loc, fname):
    """
    Convinience function for reading files in the Graves and Pitarka format
    """
    with open("/".join([loc, fname]), 'r') as f:
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

def rwm(stat_data_0_Sf,stat_data_0_Of,num_pts, dt):
#    rwm1=0
#    rwm2=0
#    rwm3=0
    t = np.arange(num_pts)*dt
    rwm1_arr=np.square((stat_data_0_Sf-stat_data_0_Of))
    rwm2_arr=np.square((stat_data_0_Sf))
    rwm3_arr=np.square((stat_data_0_Of))        
    rwm1=integrate.simps(rwm1_arr,t)
    rwm2=integrate.simps(rwm2_arr,t)
    rwm3=integrate.simps(rwm3_arr,t)

    err_rwm=rwm1/((rwm2*rwm3)**0.5)
    
    return err_rwm   

def read_stat_name(station_file):

    with open(station_file, 'r') as f:
        lines = f.readlines()
    line0=lines[0].split()
    nRec=int(line0[0])
    R=np.zeros((nRec,3))
    statnames = [] 
    for i in range(1,nRec+1):
        line_i=lines[i].split()
        R[i-1,0]=int(line_i[0])
        R[i-1,1]=int(line_i[1])
        R[i-1,2]=int(line_i[2])
        statnames.append(line_i[3])
    return nRec, R, statnames     
###################################################
source_file='../../../StatInfo/SOURCE.txt'
station_file='../../../StatInfo/STATION.txt'

nShot, S = read_source(source_file)
nRec, R, statnames = read_stat_name(station_file)
    
_, num_pts, dt, shift  = readGP_2('../../Vel_ob/Vel_ob_1','HALS.000')
t = np.arange(num_pts)*dt
############/nesi/nobackup/nesi00213/RunFolder/tdn27/rgraves/Adjoint/Syn_VMs/Kernels/#########################
fs = 1/dt
lowcut = 0.01
highcut = 0.2

fc = highcut  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(4, w, 'low')    

GV=['.090','.000','.ver']

Err=0
for ishot in range(1,nShot+1):
#for ishot in range(6,7+1):
    mainfolder='../../Vel_es/Vel_es_'+str(ishot)+'/'
    mainfolder_o='../../Vel_ob/Vel_ob_'+str(ishot)+'/'
    
    ################################
    
    for i,statname in enumerate(statnames):
    
        s0=statname+GV[0]
        s1=statname+GV[1]
        s2=statname+GV[2]
       
#        stat_data_0_S, num_pts, dt, shift  = readGP_2(mainfolder,s0)
#        stat_data_1_S, _, _, _  = readGP_2(mainfolder,s1)
#        stat_data_2_S, _, _, _  = readGP_2(mainfolder,s2)
#        
#        stat_data_0_O, num_pts, dt, shift1  = readGP_2(mainfolder_o,s0)
#        stat_data_1_O, _, _, _  = readGP_2(mainfolder_o,s1)
#        stat_data_2_O, _, _, _  = readGP_2(mainfolder_o,s2)    
#        
#        t = np.arange(num_pts)*dt
#        nt=num_pts

        stat_data_0_S  = timeseries.read_ascii(mainfolder+s0)
        stat_data_0_S  = np.multiply(signal.tukey(int(num_pts),0.1),stat_data_0_S)
        stat_data_1_S  = timeseries.read_ascii(mainfolder+s1)
        stat_data_1_S  = np.multiply(signal.tukey(int(num_pts),0.1),stat_data_1_S)   
        stat_data_2_S  = timeseries.read_ascii(mainfolder+s2)
        stat_data_2_S  = np.multiply(signal.tukey(int(num_pts),0.1),stat_data_2_S)  
        
        stat_data_0_O  = timeseries.read_ascii(mainfolder_o+s0)
        stat_data_0_O  = np.multiply(signal.tukey(int(num_pts),0.1),stat_data_0_O)
        stat_data_1_O  = timeseries.read_ascii(mainfolder_o+s1)
        stat_data_1_O  = np.multiply(signal.tukey(int(num_pts),0.1),stat_data_1_O)   
        stat_data_2_O  = timeseries.read_ascii(mainfolder_o+s2)
        stat_data_2_O  = np.multiply(signal.tukey(int(num_pts),0.1),stat_data_2_O)          

#        stat_data_0_Sf = signal.filtfilt(b, a, stat_data_0_S)
#        stat_data_1_Sf = signal.filtfilt(b, a, stat_data_1_S)
#        stat_data_2_Sf = signal.filtfilt(b, a, stat_data_2_S)
        
#        stat_data_0_Of = signal.filtfilt(b, a, stat_data_0_O)
#        stat_data_1_Of = signal.filtfilt(b, a, stat_data_1_O)
#        stat_data_2_Of = signal.filtfilt(b, a, stat_data_2_O)            

        #print('ireceiver='+str(i))
        distance=((R[i,1]-S[ishot-1,1])**2+(R[i,2]-S[ishot-1,2])**2+(R[i,0]-S[ishot-1,0])**2)**(0.5)

        if(distance>20):        
            Err=Err+ rwm(stat_data_0_S,stat_data_0_O,num_pts, dt)+rwm(stat_data_1_S,stat_data_1_O,num_pts, dt)+rwm(stat_data_2_S,stat_data_2_O,num_pts, dt)
     
f_err = open('err_obs.dat','w')
Err.astype('float').tofile(f_err)     
#print(Err)    
    
    
    
