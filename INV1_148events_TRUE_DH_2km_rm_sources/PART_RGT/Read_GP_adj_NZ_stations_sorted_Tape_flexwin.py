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
from qcore import timeseries
from scipy import integrate
import os

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

def computeFourier(accTimeSeries, dt, duration):
    #computes fourier spectra for acceleration time series
    # TODO: compute original number of points (setting to default for now) change to npts = len(accTimeSeries)
    npts = len(accTimeSeries)
    npts_FFT = int(np.ceil(duration)/dt)
    
    # compute number of points for efficient FFT
    ft_len = int(2.0 ** np.ceil(np.log(npts_FFT) / np.log(2.0)))
    if npts > ft_len:
        accTimeSeries = accTimeSeries[:ft_len]
        npts = len(accTimeSeries)
    
    # Apply hanning taper to last 5% of motion
    ntap = int(npts * 0.05) 
    accTimeSeries[npts - ntap:] *= np.hanning(ntap * 2 + 1)[ntap + 1:]
    
    # increase time series length with zeroes for FFT
    accForFFT = np.pad(accTimeSeries, (0, ft_len - len(accTimeSeries)), 'constant', constant_values=(0,0))
    ft = np.fft.rfft(accForFFT)
    
    # compute frequencies at which fourier amplitudes are computed
    ft_freq = np.arange(0, ft_len / 2 + 1) * ( 1.0 / (ft_len * dt))
    return ft, ft_freq

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

######################################
def normpdf_python(x, mu, sigma):
   return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1*(x-mu)**2/2*sigma**2)

   
##############################################
#def source_adj_1(stat_data_S,stat_data_O,num_pts,dt,wd):
#
#    """
#    Measure TauP value and construct Jp signal
#    """    
#    t = np.arange(num_pts)*dt
#   
#    ts=np.flip(-t[1:], axis=0)
#    lTime = np.concatenate((ts,t), axis=0)
#    w = normpdf_python(lTime, 0, 0.5)
#    wp = w/max(w)
#
#    stat_data_S = np.multiply(stat_data_S,wd)
#    stat_data_O = np.multiply(stat_data_O,wd)
#    
#    Dis_S = np.cumsum(stat_data_S)*dt
#    Dis_O = np.cumsum(stat_data_O)*dt
#
#    Dis_S  = np.multiply(signal.tukey(int(num_pts),0.1),Dis_S)
#    Dis_O  = np.multiply(signal.tukey(int(num_pts),0.1),Dis_O)
#
##    Dis_S = signal.detrend(Dis_S)
##    Dis_O = signal.detrend(Dis_O)
#    
#    corr=np.correlate(Dis_O,Dis_S,"full")
#    wx=np.multiply(wp,corr)
#    
#    In=np.argmax(wx)
#    TauP=lTime[In]	
#    
#    Jp_inv_norm=1/dt*np.sum(np.multiply(stat_data_S,stat_data_S))
#    Jp = TauP*np.flip(stat_data_S, axis=0)*Jp_inv_norm
#        
#    Source = Jp
#
#    return Source

##############################################
def source_adj_ncc(stat_data_Sf,stat_data_Of,num_pts, delta_T,dt):

    """
    Normalized correlation coefficient and optimal time shift
    """    
    num_delta_t = int(delta_T/dt)
    td = np.arange(-num_delta_t,num_delta_t)*dt
    ncc_array=np.zeros(len(td))
    
    for it in range(0,len(td)):
        stat_data_O_shift = np.zeros(num_pts); n_shift=int(np.abs((td[it]/dt)));
        
#        if td[it]<0:
#            stat_data_0_O_shift[n_shift:num_pts] = stat_data_0_Of[0:num_pts-n_shift]                             
#        else:
#            stat_data_0_O_shift[0:num_pts-n_shift] = stat_data_0_Of[n_shift:num_pts]         

        if td[it]<0:
            stat_data_O_shift[0:num_pts-n_shift] = stat_data_Of[n_shift:num_pts]                  
        else:
            stat_data_O_shift[n_shift:num_pts] = stat_data_Of[0:num_pts-n_shift]
        
        rwm1_arr=np.multiply(stat_data_Sf,stat_data_O_shift)
        rwm2_arr=np.square((stat_data_Sf))
        rwm3_arr=np.square((stat_data_O_shift))        
        
        rwm1=integrate.simps(rwm1_arr,t)
        rwm2=integrate.simps(rwm2_arr,t)
        rwm3=integrate.simps(rwm3_arr,t)
        
#        rwm1 = np.sum(rwm1_arr)
#        rwm2 = np.sum(rwm2_arr)
#        rwm3 = np.sum(rwm3_arr)
    
        ncc_array[it]=rwm1/((rwm2*rwm3)**0.5)
    
    ncc_max = np.max(ncc_array); id_max = np.argmax(ncc_array); td_max = td[id_max];
    
    Jp_inv_norm=1/dt*np.sum(np.multiply(stat_data_Sf,stat_data_Sf))

#    Jp = TauP*np.flip(stat_data_S, axis=0)*Jp_inv_norm
    Jp = -td_max*np.flip(stat_data_S, axis=0)*Jp_inv_norm
        
    Source = Jp

    return Source

def write_adj_source(s1,v1,mainfolder,mainfolder_source,source):
    
    with open("/".join([mainfolder, s1]), 'r') as f:
        lines = f.readlines()
    tline1=   lines[0]     
    tline2=   lines[1]
    
    filename1=mainfolder_source+v1
    print(filename1)
    fid = open(filename1,'w')
    fid.write("%s" %(tline1))
    fid.write("%s" %(tline2))
    lt=len(source)
    count=0
    while (count+1)*6<lt:
        fid.write("%10f%10f%10f%10f%10f%10f%s" %(source[count*6],source[count*6+1],source[count*6+2],source[count*6+3],source[count*6+4],source[count*6+5],'\n'))
        count+=1
    ii=lt-count*6   
    i=0
    while (i<ii):
        i+=1
        fid.write("%10f%s" %(source[lt-ii+i-1],'\n'))
    fid.close()
    return

def write_adj_source_ts(s1,v1,mainfolder,mainfolder_source,source,dt):
    #filename1=mainfolder_source+v1
    vs1=v1.split('.')
    timeseries.seis2txt(source,dt,mainfolder_source,vs1[0],vs1[1])
    return	


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
    
def rms(stat_data):
    
    num_pts=len(stat_data)    
    D = (np.sum(np.square(stat_data))/num_pts)**0.5
   
    return stat_data/D      

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
    L=t_off-t_on+2*pad
    
#    window=signal.gaussian(30, 4)
#    window1=signal.resample(window,L,axis=0, window=None)
#    window=np.linspace(1, 1, L)))
    window=np.ones((L))
    
    #x=np.arange(0,pad,1)
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
###################################################
#statnames = ['A1' ,'A2' ,'A3' ,'A4' ,'A5' ,'A6' , 'A7', 'B1' ,'B2' ,'B3' ,'B4' ,'B5' ,'B6' , 'B7','C1' ,'C2' ,'C3' ,'C4' ,'C5' ,'C6' ,'C7','D1' ,'D2' ,'D3' ,'D4' ,'D5' ,'D6' ,'D7','E1' ,'E2' ,'E3' ,'E4' ,'E5' ,'E6' ,'E7','F1' ,'F2' ,'F3' ,'F4' ,'F5' ,'F6','F7','G1' ,'G2' ,'G3' ,'G4' ,'G5' ,'G6','G7']
station_file = '../../../StatInfo/STATION.txt'    
nRec, R, statnames = read_stat_name(station_file)
#statnames=statnames[0:10]
print('statnames')
print(statnames)

GV=['.090','.000','.ver']
GV_ascii=['.x','.y','.z']

mainfolder='../../Vel_es/Vel_es_i/'
mainfolder_o='../../Vel_ob/Vel_ob_i/'
mainfolder_source='../../../AdjSims/V3.0.7-a2a_xyz/Adj-InputAscii/'
os.system('rm ../../../AdjSims/V3.0.7-a2a_xyz/Adj-InputAscii/*.*')

print(mainfolder_o)
_, num_pts, dt, shift  = readGP_2('../../Vel_ob/Vel_ob_1','CBGS.000')
num_pts=int(num_pts)
t = np.arange(num_pts)*dt
############/nesi/nobackup/nesi00213/RunFolder/tdn27/rgraves/Adjoint/Syn_VMs/Kernels/#########################
fs = 1/dt
lowcut = 0.05
#highcut = 0.05
highcut = 0.1
ndelay_T=int((3/0.1)/(dt))
delta_T=10

fc = highcut  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(4, w, 'low')

source_file='../../../StatInfo/SOURCE.txt'

nShot, S = read_source(source_file)
wr = np.loadtxt('../../../../Kernels/Iters/iter1/Dump/geo_correlation.txt')
################################
fi1=open('iShot.dat','r')
ishot=int(np.fromfile(fi1,dtype='int64'))
fi1.close()
print('ishot='+str(ishot))

#R_ishot_arr=np.loadtxt('../../../../Kernels/Iters/iter1/Dump/R_ishot_'+str(ishot)+'.txt')
R_all_arr=np.loadtxt('../../../../Kernels/index_all_ncc_gt005.txt')
R_all=R_all_arr.reshape([nRec,3,nShot])

for i,statname in enumerate(statnames):
    #print('ireceiver='+str(i))
    distance=((R[i,1]-S[ishot-1,1])**2+(R[i,2]-S[ishot-1,2])**2+(R[i,0]-S[ishot-1,0])**2)**(0.5)
    
#    source_x=np.zeros(num_pts)
#    source_y=np.zeros(num_pts)
#    source_z=np.zeros(num_pts)
    
    for k in range(0,3):
        
        source_adj=np.zeros(num_pts)
        
        s0=statname+GV[k]
        v0=statname+GV_ascii[k]
        
        if((distance<200) and (distance>0) and (R_all[i,k,ishot-1]==1)): 
            print('ireceiver='+str(i))
            wr_ij = wr[nRec*(ishot-1)+i]
   
            stat_data_0_S  = timeseries.read_ascii(mainfolder+s0)
            stat_data_0_S  = np.multiply(signal.tukey(int(num_pts),0.1),stat_data_0_S)
            stat_data_0_S_shift = np.zeros(stat_data_0_S.shape)
            stat_data_0_S_shift[0:num_pts-ndelay_T] = stat_data_0_S[ndelay_T:num_pts]     
            stat_data_0_S = stat_data_0_S_shift
        
            stat_data_0_O  = timeseries.read_ascii(mainfolder_o+s0)
            stat_data_0_O  = np.multiply(signal.tukey(int(num_pts),0.1),stat_data_0_O)
            
#            stat_data_0_O_shift = np.zeros(stat_data_0_O.shape)
#            stat_data_0_O_shift[ndelay_T:num_pts] = stat_data_0_O[0:num_pts-ndelay_T]         
#            stat_data_0_O = stat_data_0_O_shift  
            
            #stat_data_0_S = signal.detrend(stat_data_0_S)             
            #stat_data_0_O = signal.detrend(stat_data_0_O)
            
    #        stat_data_0_S = signal.filtfilt(b, a, stat_data_0_S)           
    #        stat_data_0_O = signal.filtfilt(b, a, stat_data_0_O)
            
            stat_data_0_S = butter_bandpass_filter(stat_data_0_S, lowcut, highcut, fs, order=4)       
            stat_data_0_O = butter_bandpass_filter(stat_data_0_O, lowcut, highcut, fs, order=4)             
    
            stat_data_0_O  = np.multiply(signal.tukey(int(num_pts),1.0),stat_data_0_O)
        
            stat_data_0_S = rms(stat_data_0_S)*wr_ij    
            stat_data_0_O = rms(stat_data_0_O)*wr_ij
            
            #Parameters for window    
            df=1/dt
            lt=num_pts
            pad=5
            #flexwin
            e_s_c_name = str(ishot)+'.'+s0+'.win'
            filename = '../../../../ALL_WINs/'+e_s_c_name

            t_on, t_off, td_shift, cc = read_flexwin(filename)
            tx_on=int(t_on/dt)
            tx_off=int(t_off/dt)
            #Window for isolated filter
            wd=winpad(lt,tx_off,tx_on,pad) 
            
            source_adj=source_adj_ncc(stat_data_0_S,stat_data_0_O,num_pts, delta_T,dt) 
            
        write_adj_source_ts(s0,v0,mainfolder,mainfolder_source,source_adj,dt)
        
#            if (k==0):                   
#                source_x=source_adj_ncc(stat_data_0_S,stat_data_0_O,num_pts,dt,wd)
#            if (k==1):
#                source_y=source_adj_ncc(stat_data_0_S,stat_data_0_O,num_pts,dt,wd)    
#            if (k==2):
#                source_z=source_adj_ncc(stat_data_0_S,stat_data_0_O,num_pts,dt,wd)  
#                
#    if (k==0):
#        write_adj_source_ts(s0,v0,mainfolder,mainfolder_source,source_x,dt)
#    if (k==1):
#        write_adj_source_ts(s0,v0,mainfolder,mainfolder_source,source_y,dt) 
#    if (k==2):
#        write_adj_source_ts(s0,v0,mainfolder,mainfolder_source,source_z,dt)        
       
   
    

    
    
    
