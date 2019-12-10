#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:43:17 2018

"""
#import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import signal
from scipy.signal import detrend

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
        090, 000, ver, t, stat_code
    """
    stat_data = {"090": None, "000": None, "ver":None, "t": None, "name": None}
    g=981. #cm/s^2
    stat_data["090"], num_pts, dt, shift  = readGP(loc, ".".join([stat_code, "090"]))
    stat_data["000"], num_pts, dt, shift = readGP(loc, ".".join([stat_code, "000"]))   
    stat_data["ver"], num_pts, dt, shift = readGP(loc, ".".join([stat_code, "ver"]))

    stat_data["090"], num_pts, dt = adjust_for_time_delay(stat_data["090"], dt, shift)
    stat_data["000"], num_pts, dt = adjust_for_time_delay(stat_data["000"], dt, shift)
    stat_data["ver"], num_pts, dt = adjust_for_time_delay(stat_data["ver"], dt, shift)
    
#    stat_data["090"]  = np.multiply(signal.tukey(int(num_pts),0.5),stat_data["090"])
#    stat_data["000"]  = np.multiply(signal.tukey(int(num_pts),0.5),stat_data["000"])
#    stat_data["ver"]  = np.multiply(signal.tukey(int(num_pts),0.5),stat_data["ver"])
    
#    stat_data["090"]  = detrend(stat_data["090"], type="constant")
#    stat_data["000"]  = detrend(stat_data["000"], type="constant")
#    stat_data["ver"]  = detrend(stat_data["ver"], type="constant")    

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

#statnames = ['A1','B5','C10','D15']
station_file = 'STATION.txt'    
nRec, R, statnames = read_stat_name(station_file)
statnames=statnames[0:10]
print('statnames')
print(statnames)

_, num_pts, dt, shift  = readGP('/home/user/workspace/GMPlots/Sim/Vel_ob/','HALS.000')

for i,statname in enumerate(statnames):
    #gmdata_obs = get_adjusted_stat_data('/home/user/workspace/GMPlots/Obs/Vol1/data/velBB',statname)
    gmdata_obs = get_adjusted_stat_data('/home/user/workspace/GMPlots/Sim/Vel_ob',statname)
#    gmdata_sim = get_adjusted_stat_data('/home/user/workspace/GMPlots/Sim/Vel_ob',statname)
    gmdata_sim = get_adjusted_stat_data('/home/user/workspace/GMPlots/Sim/Vel_es',statname)
    print(statname)

    ylimit=0.8
    yflimit=1.0
    vxyz='vx'
    
    gmdata_obs_vi=gmdata_obs['090']
    gmdata_sim_vi=gmdata_sim['090']
    
    fft_obs=fftpack.fft(gmdata_obs_vi)
    fft_sim=fftpack.fft(gmdata_sim_vi)
    
    fft_obs=np.abs(fft_obs)
    fft_sim=np.abs(fft_sim)
    
    sample_freq = fftpack.fftfreq(gmdata_sim['090'].size, d=0.02)
    
    plt.subplot(1,2,1)
    plt.plot(gmdata_sim['t'],gmdata_obs_vi,c='r',linestyle='solid')
    plt.plot(gmdata_sim['t'],gmdata_sim_vi,c='r',linestyle='dashed')

    plt.title('Observed vs Simulated seismograms at station '+statname, loc='center')
    plt.xlabel('Time (s)')
    plt.ylabel(vxyz+' (cm/s)')
    plt.gca().legend(('Observed','Simulated'))
#    plt.xlim([0,100])
    #ylimit = 1.1*max([ymax1,ymax4])
#    plt.plot([0,20],[-ylimit*0.50,-ylimit*0.50],c='k',linewidth=1.0)
#    plt.vlines([0,20],-ylimit*0.6,-ylimit*0.4,color='k',linewidth=1.0)
#    plt.ylim([-ylimit,ylimit])
    ax = plt.gca()
    #ax.text(0.05,0.2, '20s', horizontalalignment='left', verticalalignment='top', transform = ax.transAxes, fontsize=9.0)
    ax.axis('on')
    plt.grid()
    plt.subplot(1,2,2)
    plt.plot(sample_freq,fft_obs,c='r',linestyle='solid')
    plt.plot(sample_freq,fft_sim,c='r',linestyle='dashed')
    plt.title('FFT', loc='center') 
    plt.xlim([0,1])
#    plt.ylim([-yflimit,yflimit])
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.gca().legend(('Observed','Simulated'))
    #ax.set_yticklabels([])
    
   
    #plt.subplots_adjust(left=0.125, bottom=0.0, right=0.9, top=1.0, wspace=0.2, hspace=-0.3)
    plt.subplots_adjust(left=.1, bottom=0.0, right=2.0, top=1.0, wspace=0.2, hspace=-0.3)
    #plt.savefig(statname + '.png',dpi=200)
    plt.show()

    #ylimit=0.008
    vxyz='vy'
    
    gmdata_obs_vi=gmdata_obs['000']
    gmdata_sim_vi=gmdata_sim['000']
    
    fft_obs=fftpack.fft(gmdata_obs_vi)
    fft_sim=fftpack.fft(gmdata_sim_vi)
    
    fft_obs=np.abs(fft_obs)
    fft_sim=np.abs(fft_sim)
    
    sample_freq = fftpack.fftfreq(gmdata_sim['090'].size, d=0.02)
    
    plt.subplot(1,2,1)
    plt.plot(gmdata_sim['t'],gmdata_obs_vi,c='r',linestyle='solid')
    plt.plot(gmdata_sim['t'],gmdata_sim_vi,c='r',linestyle='dashed')

    plt.title('Observed vs Simulated seismograms at station '+statname, loc='center')
    plt.xlabel('Time (s)')
    plt.ylabel(vxyz+' (cm/s)')
    plt.gca().legend(('Observed','Simulated'))
#    plt.xlim([0,100])
    #ylimit = 1.1*max([ymax1,ymax4])
#    plt.plot([0,20],[-ylimit*0.50,-ylimit*0.50],c='k',linewidth=1.0)
#    plt.vlines([0,20],-ylimit*0.6,-ylimit*0.4,color='k',linewidth=1.0)
#    plt.ylim([-ylimit,ylimit])
    ax = plt.gca()
    #ax.text(0.05,0.2, '20s', horizontalalignment='left', verticalalignment='top', transform = ax.transAxes, fontsize=9.0)
    ax.axis('on')
    plt.grid()
    plt.subplot(1,2,2)
    plt.plot(sample_freq,fft_obs,c='r',linestyle='solid')
    plt.plot(sample_freq,fft_sim,c='r',linestyle='dashed')
    plt.title('FFT', loc='center') 
#    plt.ylim([-yflimit,yflimit])    
    plt.xlim([0,1])
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.gca().legend(('Observed','Simulated'))
    #ax.set_yticklabels([])
    
   
    #plt.subplots_adjust(left=0.125, bottom=0.0, right=0.9, top=1.0, wspace=0.2, hspace=-0.3)
    plt.subplots_adjust(left=.1, bottom=0.0, right=2.0, top=1.0, wspace=0.2, hspace=-0.3)
    #plt.savefig(statname + '.png',dpi=200)
    plt.show()
    
    #ylimit=0.008
    vxyz='vz'
    
    gmdata_obs_vi=gmdata_obs['ver']
    gmdata_sim_vi=gmdata_sim['ver']
    
    fft_obs=fftpack.fft(gmdata_obs_vi)
    fft_sim=fftpack.fft(gmdata_sim_vi)
    
    fft_obs=np.abs(fft_obs)
    fft_sim=np.abs(fft_sim)
    
    sample_freq = fftpack.fftfreq(gmdata_sim['090'].size, d=0.02)
    
    plt.subplot(1,2,1)
    plt.plot(gmdata_sim['t'],gmdata_obs_vi,c='r',linestyle='solid')
    plt.plot(gmdata_sim['t'],gmdata_sim_vi,c='r',linestyle='dashed')

    plt.title('Observed vs Simulated seismograms at station '+statname, loc='center')
    plt.xlabel('Time (s)')
    plt.ylabel(vxyz+' (cm/s)')
    plt.gca().legend(('Observed','Simulated'))
#    plt.xlim([0,100])
    #ylimit = 1.1*max([ymax1,ymax4])
#    plt.plot([0,20],[-ylimit*0.50,-ylimit*0.50],c='k',linewidth=1.0)
#    plt.vlines([0,20],-ylimit*0.6,-ylimit*0.4,color='k',linewidth=1.0)
#    plt.ylim([-ylimit,ylimit])
    ax = plt.gca()
    #ax.text(0.05,0.2, '20s', horizontalalignment='left', verticalalignment='top', transform = ax.transAxes, fontsize=9.0)
    ax.axis('on')
    plt.grid()
    plt.subplot(1,2,2)
    plt.plot(sample_freq,fft_obs,c='r',linestyle='solid')
    plt.plot(sample_freq,fft_sim,c='r',linestyle='dashed')
    plt.title('FFT', loc='center') 
#    plt.ylim([-yflimit,yflimit])    
    plt.xlim([0,1])
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.gca().legend(('Observed','Simulated'))
    #ax.set_yticklabels([])
    
   
    #plt.subplots_adjust(left=0.125, bottom=0.0, right=0.9, top=1.0, wspace=0.2, hspace=-0.3)
    plt.subplots_adjust(left=.1, bottom=0.0, right=2.0, top=1.0, wspace=0.2, hspace=-0.3)
    #plt.savefig(statname + '.png',dpi=200)
    plt.show()