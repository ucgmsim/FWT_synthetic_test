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

def rms(stat_data):

    num_pts=len(stat_data)
    D = (np.sum(np.square(stat_data))/num_pts)**0.5

    return stat_data/D

def rwm(stat_data_0_Sf,stat_data_0_Of,num_pts, dt):

    t = np.arange(num_pts)*dt
    rwm1_arr=np.square((stat_data_0_Sf-stat_data_0_Of))
    rwm2_arr=np.square((stat_data_0_Sf))
    rwm3_arr=np.square((stat_data_0_Of))        
    rwm1=integrate.simps(rwm1_arr,t)
    rwm2=integrate.simps(rwm2_arr,t)
    rwm3=integrate.simps(rwm3_arr,t)

    err_rwm=rwm1/((rwm2*rwm3)**0.5)
    
    return err_rwm   

def read_flexwin(filename):
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    if(len(lines)>2):
        line2=lines[2].split()
        t_on=float(line2[1])
        t_off=float(line2[2])
        td_shift=float(line2[3])
        cc=float(line2[4])
        
    else:
        t_on = 0; t_off = 0; td_shift = 0; cc = 0;       
    
    return t_on, t_off, td_shift, cc    

def winpad(lt,t_off,t_on,pad):
    """
    Sin taper window
    """    
    #pad=5
    if(t_on<10):
        t_on=10
    if(t_off>lt-10):
        t_off=lt-10

    L=t_off-t_on+2*pad
    
    window=np.ones((L))

    x=np.linspace(0, np.pi/2, pad)
    sinx=np.sin(x)
    window[0:pad] = sinx
    window[L-pad:L] = sinx[::-1]    
    print('lt='+str(lt))    
    ar1=np.zeros((t_on-pad))
    ar2=np.zeros((lt-t_off-pad))
    window_pad0 = np.concatenate((ar1,window))
    window_pad = np.concatenate((window_pad0,ar2))    
    
    return window_pad 

def rwm_wd(stat_data_0_Sf,stat_data_0_Of,num_pts, dt,wd):
#    rwm1=0
#    rwm2=0
#    rwm3=0
    t = np.arange(num_pts)*dt
    stat_data_0_Sf = np.multiply(stat_data_0_Sf,wd)
    stat_data_0_Of = np.multiply(stat_data_0_Of,wd)

    rwm1_arr=np.square((stat_data_0_Sf-stat_data_0_Of))
    rwm2_arr=np.square((stat_data_0_Sf))
    rwm3_arr=np.square((stat_data_0_Of))

    rwm1=integrate.simps(rwm1_arr,t)
    rwm2=integrate.simps(rwm2_arr,t)
    rwm3=integrate.simps(rwm3_arr,t)

#    rwm1 = np.sum(np.cumsum(rwm1_arr[wd1:wd2])*dt)
#    rwm2 = np.sum(np.cumsum(rwm2_arr[wd1:wd2])*dt)
#    rwm3 = np.sum(np.cumsum(rwm3_arr[wd1:wd2])*dt)    

    err_rwm=rwm1/((rwm2*rwm3)**0.5)

#    if(err_rwm>4):
#        err_rwm=0

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

def time_shift_emod3d(data,delay_Time,dt):
    n_pts = len(data)
    ndelay_Time = int(delay_Time/(dt))
    data_shift = np.zeros(data.shape)
    data_shift[0:n_pts-ndelay_Time] = data[ndelay_Time:n_pts]
    return data_shift
###################################################
source_file='../../../StatInfo/SOURCE.txt'
station_file='../../../StatInfo/STATION.txt'

nShot, S = read_source(source_file)
nRec, R, statnames = read_stat_name(station_file)
    
_, num_pts, dt, shift  = readGP_2('../../Vel_ob_new_V2/Vel_ob_1','CBGS.000')
num_pts = int(num_pts)
t = np.arange(num_pts)*dt
############/nesi/nobackup/nesi00213/RunFolder/tdn27/rgraves/Adjoint/Syn_VMs/Kernels/#########################
fs = 1/dt
lowcut = 0.05
#highcut = 0.05
highcut = 0.1
#ndelay_T=int((3/0.1)/(dt))
delta_T=10
flo=0.1
delay_Time=(3/flo)

fc = highcut  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(4, w, 'low')

GV=['.090','.000','.ver']

Err=0
count=0

#R_all_arr=np.loadtxt('../../index_all_ncc_gt005.txt')
#R_all_arr=np.loadtxt('../../index_all_ncc_gt005_V2.txt')
#R_all_arr=np.loadtxt('../../index_all_ncc_pyflex_gt05_V2.txt')
R_all_arr=np.loadtxt('../../index_all_ncc_pyflex_gt05_V2_excluded.txt')

R_all=R_all_arr.reshape([nRec,3,nShot])

#for ishot in range(1,nShot+1):
#for ishot in range(95,104+1):
#ishot_arr=[28, 120, 132, 139, 143]
#ishot_arr=[1, 4, 5, 28, 69]
#ishot_arr=[1, 5,28, 140, 141,143]
#ishot_arr=[1, 5, 28, 34, 140, 141,142]
#ishot_arr=[37, 55, 98, 120, 69]
#ishot_arr=[4, 5, 28, 139, 142, 37, 55, 98, 120, 69]
#ishot_arr=[1, 5, 28, 34, 140, 141,142, 4, 29, 133, 139, 143]
ishot_arr=[1, 5, 28, 37, 55, 69, 98, 120]
#ishot_arr=[45, 69, 132, 142, 143]

for ishot_id in range(0,len(ishot_arr)):
    ishot=ishot_arr[ishot_id]

    mainfolder='../../Vel_opt/Vel_ob_'+str(ishot)+'/'
    mainfolder_o='../../Vel_ob_new_V2/Vel_ob_'+str(ishot)+'/'
    
    ################################
#    R_ishot_arr=np.loadtxt('Dump/R_ishot_'+str(ishot)+'.txt')
    if ((S[ishot-1,2]<20) and (np.sum(R_all[:,:,ishot-1])>3)):
#    if ((S[ishot-1,2]<20) and (np.sum(R_all[:,:,ishot-1])>9)):
#    if ((S[ishot-1,2]<20) and (np.sum(R_all[:,:,ishot-1])>6)):
    
        for i,statname in enumerate(statnames):
        
            distance=((R[i,1]-S[ishot-1,1])**2+(R[i,2]-S[ishot-1,2])**2+(R[i,0]-S[ishot-1,0])**2)**(0.5)      
            
            for k in range(0,3): 
                s0=statname+GV[k]           
                
                if ((distance<200) and (distance>0) and (R_all[i,k,ishot-1]==1)): 
                    
                    stat_data_0_S_org  = timeseries.read_ascii(mainfolder+s0)
                    stat_data_0_S = time_shift_emod3d(stat_data_0_S_org,delay_Time,dt)            
                    
                    stat_data_0_S  = np.multiply(signal.tukey(int(num_pts),0.1),stat_data_0_S)
                           
                    stat_data_0_O  = timeseries.read_ascii(mainfolder_o+s0)
                    stat_data_0_O  = np.multiply(signal.tukey(int(num_pts),0.1),stat_data_0_O)                    
                    #stat_data_0_S = signal.detrend(stat_data_0_S)             
                    #stat_data_0_O = signal.detrend(stat_data_0_O)
                    
            #        stat_data_0_S = signal.filtfilt(b, a, stat_data_0_S)           
            #        stat_data_0_O = signal.filtfilt(b, a, stat_data_0_O)
                    
                    stat_data_0_S = butter_bandpass_filter(stat_data_0_S, lowcut, highcut, fs, order=4)       
                    stat_data_0_O = butter_bandpass_filter(stat_data_0_O, lowcut, highcut, fs, order=4)             
            
                    #stat_data_0_O  = np.multiply(signal.tukey(int(num_pts),1.0),stat_data_0_O)
                
            #        stat_data_0_S = rms(stat_data_0_S)   
            #        stat_data_0_O = rms(stat_data_0_O)
           
                    #Parameters for window    
                    df=1/dt
                    lt=num_pts
                    pad=5
                    #flexwin
                    e_s_c_name = str(ishot)+'.'+s0+'.win'
                    #filename = '../../ALL_WINs_pyflex_V2/'+e_s_c_name
                    filename = '../../ALL_WINs_pyflex_temp/'+e_s_c_name
        
                    t_on, t_off, td_shift, cc = read_flexwin(filename)
                    tx_on=int(t_on/dt)
                    tx_off=int(t_off/dt)
                    #Window for isolated filter
                    wd=winpad(lt,tx_off,tx_on,pad) 
                    
                    Err=Err+ rwm_wd(stat_data_0_S,stat_data_0_O,num_pts, dt,wd)
                    
           #Err=Errr rwm(stat_data_0_S,stat_data_0_O,num_pts, dt)+rwm(stat_data_1_S,stat_data_1_O,num_pts, dt)+rwm(stat_data_2_S,stat_data_2_O,num_pts, dt)
    else:
         print('No calculation for source '+str(ishot))
         count=count+1
             
f_err = open('err_opt.dat','w')
Err.astype('float').tofile(f_err)     
print('Removing '+str(count)+ ' sources ')    
    
    
    
