#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:43:17 2018
    
@author: robin
"""
#import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import signal

def readGP(loc, fname):
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

def adjust_for_time_delay(ts, dt, shift):
    """
        ts: time series data
    """
    t0_index = int(shift/dt)
    if t0_index == 0:
        num_pts = ts.size
    elif t0_index > 0:
        ts = np.concatenate((np.zeros(t0_index), ts))
        num_pts = ts.size
    elif t0_index <0:
        ts = ts[np.abs(t0_index):]
        num_pts = ts.size

    return ts, num_pts, dt

def get_adjusted_stat_data(loc, stat_code):
    """
    Time series data is adjust for time shift

    returns a dictionary with components:
        000, 090, ver, t, stat_code
    """
    stat_data = {"000": None, "090": None, "ver":None, "t": None, "name": None}
    g=981. #cm/s^2
    stat_data["000"], num_pts, dt, shift  = readGP(loc, ".".join([stat_code, "000"]))
    stat_data["090"], num_pts, dt, shift = readGP(loc, ".".join([stat_code, "090"]))   
    stat_data["ver"], num_pts, dt, shift = readGP(loc, ".".join([stat_code, "ver"]))


    stat_data["000"], num_pts, dt = adjust_for_time_delay(stat_data["000"], dt, shift)
    stat_data["090"], num_pts, dt = adjust_for_time_delay(stat_data["090"], dt, shift)
    stat_data["ver"], num_pts, dt = adjust_for_time_delay(stat_data["ver"], dt, shift)

    t = np.arange(num_pts)*dt
    stat_data["t"] = t

    stat_data["name"]=stat_code
    return stat_data

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

def read_e_stat_name(e_station_file):

    with open(e_station_file, 'r') as f:
        lines = f.readlines()
    nRec1=len(lines) 
    statnames1 = [] 
    for i in range(0,nRec1,1):
        line_i=lines[i].split()
        statnames1.append(line_i[0])
    return nRec1, statnames1

def redefy_stat_name(new_station_file, e_station_file, station_file):
    nRec1, statnames1 = read_e_stat_name(e_station_file)

    
    with open(station_file, 'r') as f:
        lines = f.readlines()
    line0=lines[0].split()
    nRec=int(line0[0])
    
    nRec2=0
    for i in range(1,nRec+1):
        line_i=lines[i].split()
        if(line_i[3] in statnames1):
#            fid.write("%s"%(lines[i]))      
            nRec2=nRec2+1
            
    fid=open(new_station_file,'w')
    fid.write("%s\n"%(str(int(nRec2))))   
    
    for i in range(1,nRec+1):
        line_i=lines[i].split()
        if(line_i[3] in statnames1):
            fid.write("%s"%(lines[i]))      
    
    return
##statnames = ['A1','B5','C10','D15']
station_file = 'STATION.txt'  
e_station_file = 'tmp.txt'  
new_station_file = 'NEW_STATION.txt'
redefy_stat_name(new_station_file, e_station_file, station_file)
nRec, R, statnames = read_stat_name(new_station_file)
statnames=statnames[0:4]
print('statnames')
print(statnames)

_, num_pts, dt, shift  = readGP_2('/home/user/workspace/GMPlots/Sim/Vel_es/','HALS.000')
t = np.arange(num_pts)*dt

############/nesi/nobackup/nesi00213/RunFolder/tdn27/rgraves/Adjoint/Syn_VMs/Kernels/#########################
fs = 1/dt
lowcut = 0.0
highcut = 0.1

fc = highcut  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(5, w, 'low')

for i,statname in enumerate(statnames):
    #gmdata_sim = get_adjusted_stat_data('/home/user/workspace/GMPlots/Sim/Vel_es',statname)
    gmdata_sim = get_adjusted_stat_data('/home/user/workspace/GMPlots/Sim/Vel_ob',statname)
    print(statname)
    #ymax4 = max(abs(gmdata_sim['000']))
    #ymax5 = max(abs(gmdata_sim['090']))
    ymax4 = max(abs(gmdata_sim['090']))
    ymax5 = max(abs(gmdata_sim['000']))    
    ymax6 = max(abs(gmdata_sim['ver']))
    
    #ylimit = 1.05*max([ymax4,ymax5,ymax6])
    #ylimit=0.08
    ylimit=0.02
    yflimit=8
    
#    fft_0=fftpack.fft(gmdata_sim['000'])
#    fft_1=fftpack.fft(gmdata_sim['090'])
    fft_0=fftpack.fft(gmdata_sim['090'])
    fft_1=fftpack.fft(gmdata_sim['000'])    
    fft_2=fftpack.fft(gmdata_sim['ver'])
    
    fft_0=np.abs(fft_0)
    fft_1=np.abs(fft_1)
    fft_2=np.abs(fft_2)
    
    sample_freq = fftpack.fftfreq(gmdata_sim['000'].size, d=0.02)
    
    plt.subplot(1,2,1)
#    plt.plot(gmdata_sim['t'],gmdata_sim['000'],c='r')
#    plt.plot(gmdata_sim['t'],gmdata_sim['090'],c='g')
    plt.plot(gmdata_sim['t'],gmdata_sim['090'],c='r')
    plt.plot(gmdata_sim['t'],gmdata_sim['000'],c='g')    
    plt.plot(gmdata_sim['t'],gmdata_sim['ver'],c='b')
    plt.title('Seismograms at station '+statname, loc='center')
    plt.xlabel('Time (s)')
    plt.gca().legend(('vx','vy','vz'))
    plt.ylabel('Velocity (cm/s)')
    plt.xlim([0,100])
#    plt.ylim([-ylimit,ylimit])
    ax = plt.gca()
    plt.grid()    
    

    plt.subplot(1,2,2)
    plt.plot(sample_freq,fft_0,c='r')
    plt.plot(sample_freq,fft_1,c='g')
    plt.plot(sample_freq,fft_2,c='b')
    plt.title('FFT', loc='center') 
    plt.xlim([0,2])
#    plt.ylim([-yflimit,yflimit])
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.gca().legend(('vx','vy','vz'))
    #ax.set_yticklabels([])
   
    #plt.subplots_adjust(left=0.125, bottom=0.0, right=0.9, top=1.0, wspace=0.2, hspace=-0.3)
    plt.subplots_adjust(left=.1, bottom=0.0, right=2.0, top=1.0, wspace=0.2, hspace=-0.3)
    #plt.savefig(statname + '.png',dpi=200)
    plt.show()

    stat_data_0_Of = signal.filtfilt(b, a, gmdata_sim['090'])
    stat_data_1_Of = signal.filtfilt(b, a, gmdata_sim['000'])
    stat_data_2_Of = signal.filtfilt(b, a, gmdata_sim['ver'])

    plt.subplot(1,2,1)
#    plt.plot(gmdata_sim['t'],gmdata_sim['000'],c='r')
#    plt.plot(gmdata_sim['t'],gmdata_sim['090'],c='g')
    plt.plot(gmdata_sim['t'],stat_data_0_Of,c='r')
    plt.plot(gmdata_sim['t'],stat_data_1_Of,c='g')    
    plt.plot(gmdata_sim['t'],stat_data_2_Of,c='b')
    plt.title('Filtered Seismograms at station '+statname, loc='center')
    plt.xlabel('Time (s)')
    plt.gca().legend(('vx','vy','vz'))
    plt.ylabel('Velocity (cm/s)')
    plt.xlim([0,100])
#    plt.ylim([-ylimit,ylimit])
    ax = plt.gca()
    plt.grid()    
    
    fft_0=fftpack.fft(stat_data_0_Of)
    fft_1=fftpack.fft(stat_data_1_Of)    
    fft_2=fftpack.fft(stat_data_2_Of)
    
    fft_0=np.abs(fft_0)
    fft_1=np.abs(fft_1)
    fft_2=np.abs(fft_2)    

    plt.subplot(1,2,2)
    plt.plot(sample_freq,fft_0,c='r')
    plt.plot(sample_freq,fft_1,c='g')
    plt.plot(sample_freq,fft_2,c='b')
    plt.title('FFT', loc='center') 
    plt.xlim([0,2])
#    plt.ylim([-yflimit,yflimit])
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.gca().legend(('vx','vy','vz'))
    #ax.set_yticklabels([])
   
    #plt.subplots_adjust(left=0.125, bottom=0.0, right=0.9, top=1.0, wspace=0.2, hspace=-0.3)
    plt.subplots_adjust(left=.1, bottom=0.0, right=2.0, top=1.0, wspace=0.2, hspace=-0.3)
    #plt.savefig(statname + '.png',dpi=200)
    plt.show()
