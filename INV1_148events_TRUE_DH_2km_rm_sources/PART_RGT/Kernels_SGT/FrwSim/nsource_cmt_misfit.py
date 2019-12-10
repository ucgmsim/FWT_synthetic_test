#!/usr/bin/env python
# Generated with SMOP  0.41
#from libsmop import *
#import os
"""
Created on Mon Dec  9 17:20:52 2019

@author: user

This script will calculate the CMT solution for an event directly from SGT data base and run simulation 
with the new CMT solution, calculate the misfit between the new simulated vs observed data and compare with 
the old misfit to see if the new CMT solution is actually better than the one given from GeoNet

"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy import fftpack
from scipy import signal
from qcore import timeseries
from scipy import integrate
from scipy.signal import butter, lfilter
from scipy.signal import resample

from numpy.linalg import inv
#import pdb; 
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

def Vsp_read(nx,ny,nz,snap_file):
    fid=open(snap_file,'r')
    sek=np.fromfile(fid, dtype='<f4')
    #print(sek.shape)
    Vp=np.reshape(sek,[ny,nz,nx])

    return Vp

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

def read_source_new(source_file):
    
    with open(source_file, 'r') as f:
        lines = f.readlines()    
    line0=lines[0].split()
    nShot=int(line0[0])
    S=np.zeros((nShot,3))
    sNames=[]
    for i in range(1,nShot+1):
        line_i=lines[i].split()
        S[i-1,0]=line_i[0]
        S[i-1,1]=line_i[1]
        S[i-1,2]=line_i[2]
        sNames.append(line_i[3])
    
    return nShot, S, sNames    

def time_shift_emod3d(data,delay_Time,dt):
    n_pts = len(data)
    ndelay_Time = int(delay_Time/(dt))
    data_shift = np.zeros(data.shape)
    data_shift[0:n_pts-ndelay_Time] = data[ndelay_Time:n_pts]
    return data_shift
      
def rwm_wd(stat_data_0_Sf,stat_data_0_Of,num_pts, dt,wd):

    t = np.arange(num_pts)*dt
    stat_data_0_Sf_wd = np.multiply(stat_data_0_Sf,wd)
    stat_data_0_Of_wd = np.multiply(stat_data_0_Of,wd)

    rwm1_arr=np.square((stat_data_0_Sf_wd-stat_data_0_Of_wd))
    rwm2_arr=np.square((stat_data_0_Sf_wd))
    rwm3_arr=np.square((stat_data_0_Of_wd))

    rwm1=integrate.simps(rwm1_arr,t)
    rwm2=integrate.simps(rwm2_arr,t)
    rwm3=integrate.simps(rwm3_arr,t)

#    rwm1 = np.sum(np.cumsum(rwm1_arr)*dt)
#    rwm2 = np.sum(np.cumsum(rwm2_arr)*dt)
#    rwm3 = np.sum(np.cumsum(rwm3_arr)*dt)    

    err_rwm=rwm1/((rwm2*rwm3)**0.5)

    return err_rwm
    
# Read_Dev_strain_ex2.m
nx=65
ny=65
#nz=75
nz=40
nts=2000
nt0=400
nxyz=nx*ny*nz

nt=2000; dt=0.08;
t = np.arange(nt)*dt

fs = 1/dt
lowcut = 0.05
#highcut = 0.4
highcut = 0.1

fc = highcut  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(4, w, 'low')

flo=0.1
delay_Time=(3/flo)

station_file = '../../../StatInfo/STATION.txt'  
nRec, R, statnames = read_stat_name(station_file)

source_file = '../../../StatInfo/SOURCE.txt'
nShot, S, sNames = read_source_new(source_file)

#statnames_RGT = ['CACS', 'REHS']
statnames_RGT = ['CACS', 'CBGS', 'CRLZ', 'REHS', 'NNBS']
#statnames_RGT = ['CACS']

#ishot_arr=[1, 5, 28, 37, 55, 98, 120, 69]
ishot_arr=[5]

R_Time_record_arr = np.loadtxt('../R_Time_record_148s_dh_2km.txt') 
R_Time_record = R_Time_record_arr.reshape([2,nShot,nRec])

GV=['.090','.000','.ver']

Err=0

for ishot_id in range(0,len(ishot_arr)):
    ishot=ishot_arr[ishot_id]
    
#    os.system('cp Vel_ob_new_V2/Vel_ob_'+str(ishot)+'/*.* /home/user/workspace/GMPlots/Sim/Vel_ob/')
#    os.system('cp Vel_ob_ref_01Hz/Vel_ob_'+str(ishot)+'/*.* /home/user/workspace/GMPlots/Sim/Vel_es/')    

#    mainfolder='/home/user/workspace/GMPlots/Sim/Vel_es/'
#    mainfolder_o='/home/user/workspace/GMPlots/Sim/Vel_ob/'

    #mainfolder='Vel_opt/Vel_ob_'+str(ishot)+'/'
    mainfolder='../Vel_opt/Vel_opt_'+str(ishot)+'/'
#    mainfolder_o='../../Kernels/Vel_ob_new_V2/Vel_ob_'+str(ishot)+'/'
    mainfolder_o='../../../Kernels/Vel_ob_ref/Vel_ob_'+str(ishot)+'/'
       
    for i,statname in enumerate(statnames):
        if(statname in statnames_RGT):
            #distance=((R[i,1]-S[ishot-1,1])**2+(R[i,2]-S[ishot-1,2])**2+(R[i,0]-S[ishot-1,0])**2)**(0.5)
            print(statname)
            for k in range(0,3):
                s0=statname+GV[k]
    
                stat_data_0_S_org  = timeseries.read_ascii(mainfolder+s0)
                stat_data_0_S = time_shift_emod3d(stat_data_0_S_org,delay_Time,dt)
                stat_data_0_S  = np.multiply(signal.tukey(int(nt),0.1),stat_data_0_S)
    
                stat_data_0_O  = timeseries.read_ascii(mainfolder_o+s0)
                stat_data_0_O  = np.multiply(signal.tukey(int(nt),0.1),stat_data_0_O)
                #stat_data_0_S = signal.detrend(stat_data_0_S)             
                #stat_data_0_O = signal.detrend(stat_data_0_O)
    
        #        stat_data_0_S = signal.filtfilt(b, a, stat_data_0_S)           
        #        stat_data_0_O = signal.filtfilt(b, a, stat_data_0_O)
    
                stat_data_0_S = butter_bandpass_filter(stat_data_0_S, lowcut, highcut, fs, order=4)
                stat_data_0_O = butter_bandpass_filter(stat_data_0_O, lowcut, highcut, fs, order=4)
    
                wd=np.zeros(stat_data_0_S.shape)    
                wd[int(R_Time_record[0,ishot-1,i]/dt):int(R_Time_record[1,ishot-1,i]/dt)+1]=1     
                wd=np.ones(wd.shape)
    
                #stat_data_0_S = np.multiply(stat_data_0_S,wd)
                #stat_data_0_O = np.multiply(stat_data_0_O,wd) 
                
                Err = Err+rwm_wd(stat_data_0_S,stat_data_0_O,nt, dt,wd)
                           
print('Err='+str(Err))
f_err = open('err_cmt.dat','w')            
            

