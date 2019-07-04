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
from qcore import timeseries
from scipy import integrate
from scipy.signal import butter, lfilter
import sys

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

def read_source(source_file):

    with open(source_file, 'r') as f:
        lines = f.readlines()
    line0=lines[0].split()
    nShot=int(line0[0])
    S=np.zeros((nShot,3))
    for i in range(1,nShot+1):
        line_i=lines[i].split()
        S[i-1,0]=int(line_i[0])
        S[i-1,1]=int(line_i[1])
        S[i-1,2]=int(line_i[2])

    return nShot, S

#def butter_bandpass(lowcut, highcut, fs, order):
#    nyq = 0.5 * fs
#    low = lowcut / nyq
#    high = highcut / nyq
#    b, a = butter(order, [low, high], btype='band')
#    return b, a
#
#
#def butter_bandpass_filter(data, lowcut, highcut, fs, order):
#    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#    y = lfilter(b, a, data)
#    return y
def rms(stat_data):

    num_pts=len(stat_data)
    D = (np.sum(np.square(stat_data))/num_pts)**0.5

    return stat_data/D

def ncc(stat_data_0_Sf,stat_data_0_Of,num_pts, delta_T,dt):
    """
    Normalized correlation coefficient
    """    
#    rwm1=0
#    rwm2=0
#    rwm3=0
    num_delta_t = int(delta_T/dt);    td = np.arange(-num_delta_t,num_delta_t)*dt;
    t = np.arange(num_pts)*dt
    ncc_array=np.zeros(len(td))
    
    for it in range(0,len(td)):
        stat_data_0_S_shift = np.zeros(num_pts); n_shift=int(np.abs((td[it]/dt)));
        
        if td[it]<0:
            stat_data_0_S_shift[n_shift:num_pts] = stat_data_0_Sf[0:num_pts-n_shift]                             
        else:
            stat_data_0_S_shift[0:num_pts-n_shift] = stat_data_0_Sf[n_shift:num_pts]                     
        
        rwm1_arr=np.multiply(stat_data_0_S_shift,stat_data_0_Of)
        rwm2_arr=np.square((stat_data_0_S_shift))
        rwm3_arr=np.square((stat_data_0_Of))        
        
        rwm1=integrate.simps(rwm1_arr,t)
        rwm2=integrate.simps(rwm2_arr,t)
        rwm3=integrate.simps(rwm3_arr,t)
    
        ncc_array[it]=rwm1/((rwm2*rwm3)**0.5)
    
    ncc_max = np.max(ncc_array); id_max = np.argmax(ncc_array); td_max = td[id_max];
    
    return ncc_max, td_max   

#####################################
orig_stdout = sys.stdout
f = open('out.txt', 'w')
sys.stdout = f

#station_file = '/home/user/workspace/GMPlots/Christchurch_Events/INV1_4events_TRUE/Kernels/STATION.txt'  
station_file = 'STATION.txt'      
nRec, R, statnames = read_stat_name(station_file)

source_file='SOURCE.txt'
nShot, S = read_source(source_file)
#nShot=4;ishot=1;

R_arr=np.loadtxt('R_all_148s.txt')
R_all=np.reshape(R_arr,[nShot,nRec])

#statnames=statnames[0:20]
print('statnames')
print(statnames)

num_pts=10000; dt=0.02;
#num_pts=5000; dt=0.02;
#_, num_pts, dt, shift  = readGP_2('/home/user/workspace/GMPlots/Sim/Vel_es/','HALS.000')

t = np.arange(num_pts)*dt

############/nesi/nobackup/nesi00213/RunFolder/tdn27/rgraves/Adjoint/Syn_VMs/Kernels/#########################
fs = 1/dt
lowcut = 0.01
highcut = 0.1

fc = highcut  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(4, w, 'low')

GV=['.090','.000','.ver']

for ishot in range(1,nShot+1):
    if (S[ishot-1,2]<40):
        ncc_R=np.zeros((nRec,3)); td_max_R=np.zeros((nRec,3));
        delta_T=5
        #delta_T=2.5
        R_ishot_new = R_all[ishot-1,:]
        
        for i,statname in enumerate(statnames):
            if (R_all[ishot-1,i]==1):
                for k in range(0,3):
                    s0=statname+GV[k]
                    gmdata_obs_vi_O = timeseries.read_ascii('/home/user/workspace/GMPlots/Christchurch_Events/INV1_148events_TRUE/Kernels/Vel_ob/Vel_ob_'+str(ishot)+'/'+s0)
                    gmdata_sim_vi_O = timeseries.read_ascii('/home/user/workspace/GMPlots/Christchurch_Events/INV1_148events_TRUE/Kernels/Vel_ob_ref/Vel_ob_'+str(ishot)+'/'+s0)
                    
                    print(statname)
                
                    ylimit=0.0002
                    yflimit=0.03
                    
        #            vxyz='vx'
                    vxyz=GV[k]
                           
                    gmdata_obs_vi = signal.filtfilt(b, a, gmdata_obs_vi_O)               
                    gmdata_sim_vi = signal.filtfilt(b, a, gmdata_sim_vi_O)
            #        gmdata_obs_vi = butter_bandpass_filter(gmdata_obs_vi_O, lowcut, highcut, fs, order=4)        
            #        gmdata_sim_vi = butter_bandpass_filter(gmdata_sim_vi_O, lowcut, highcut, fs, order=4)  
                    
    #                gmdata_sim_vi = gmdata_sim_vi*np.max(np.abs(gmdata_obs_vi))/np.max(np.abs(gmdata_sim_vi))
                    gmdata_obs_vi = rms(gmdata_obs_vi)
                    gmdata_sim_vi = rms(gmdata_sim_vi)               
                    
                    fft_obs=fftpack.fft(gmdata_obs_vi)
                    fft_sim=fftpack.fft(gmdata_sim_vi)
                    
                    fft_obs=np.abs(fft_obs)
                    fft_sim=np.abs(fft_sim)
                    
                    sample_freq = fftpack.fftfreq(gmdata_obs_vi.size, d=0.02)
                                   
                    ncc_R[i,k], td_max_R[i,k] = ncc(gmdata_sim_vi,gmdata_obs_vi,num_pts, delta_T,dt)
                    
                if (np.sum(ncc_R[i,:])/3<0.2):
                    td_max_R[i,:]=0
                    R_ishot_new[i]=0            
                    print('remove stat='+statname+'from source '+str(ishot))
                    
#        plt.figure(figsize=(10,2.5))       
#        plt.subplot(1,2,1)
#        plt.plot(ncc_R[:,0],c='k',linestyle='solid')
#        plt.xlabel('NCC')
#        plt.grid()
#        plt.subplot(1,2,2)
#        plt.plot(td_max_R[:,0],c='k',linestyle='solid')
#        plt.xlabel('Td_max')
#        plt.ylabel('Time [s]')
#        plt.show()        
                
        np.savetxt('R_ishot/R_ishot_'+str(ishot)+'.txt', R_ishot_new)

sys.stdout = orig_stdout
f.close()